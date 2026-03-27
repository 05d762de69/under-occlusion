from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Record:
    category: str
    contour_u: np.ndarray  # Nx2 in [0,1] (unit coords)
    img_id: Optional[int] = None


def _as_2col_array(x) -> np.ndarray:
    arr = np.asarray(x)
    arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    if arr.shape[1] == 2:
        out = arr
    elif arr.shape[0] == 2:
        out = arr.T
    else:
        raise ValueError(f"Contour must be Nx2 or 2xN. Got shape {arr.shape}")

    return out.astype(np.float64, copy=False)


def _to_str(x) -> str:
    try:
        s = str(np.squeeze(x))
        # MATLAB sometimes gives "['monkey']" style. This keeps it simple.
        return s.strip()
    except Exception:
        return ""


def _try_load_classic_mat(path: Path) -> Optional[dict]:
    try:
        from scipy.io import loadmat
    except Exception as e:
        raise RuntimeError("scipy is required to read classic .mat files") from e

    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception:
        return None


def _try_load_v73_mat(path: Path):
    try:
        import h5py
    except Exception:
        return None
    try:
        return h5py.File(path, "r")
    except Exception:
        return None


def _extract_records_classic(d: dict) -> List[Record]:
    if "records" not in d:
        raise ValueError("master_lib.mat must contain variable 'records'.")

    recs = d["records"]
    rec_list = np.ravel(recs) if isinstance(recs, np.ndarray) else np.array([recs], dtype=object)

    out: List[Record] = []
    for r in rec_list:
        # Expect fields: category, contour, (optional) img_id
        if not hasattr(r, "category") or not hasattr(r, "contour"):
            continue

        cat = _to_str(r.category)
        contour_u = _as_2col_array(r.contour)

        img_id = None
        if hasattr(r, "img_id"):
            try:
                img_id = int(np.squeeze(r.img_id))
            except Exception:
                img_id = None

        out.append(Record(category=cat, contour_u=contour_u, img_id=img_id))

    if not out:
        raise ValueError("No valid records found. Expected fields: category, contour.")

    return out


def _read_h5_str(obj) -> str:
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="ignore")
    arr = np.array(obj)
    arr = np.squeeze(arr)
    if arr.dtype.kind in {"S", "U"}:
        return str(arr)
    if arr.dtype.kind in {"u", "i"}:
        try:
            return "".join(chr(int(c)) for c in arr.flatten())
        except Exception:
            return ""
    return ""


def _extract_records_v73(f) -> List[Record]:
    """
    Robust v7.3 (HDF5) reader for MATLAB struct arrays stored as object references.

    Common MATLAB v7.3 layouts:
      - f['records'] is a dataset of object references, each ref -> a struct-like group
      - fields like category/contour/img_id may themselves be references

    This function dereferences those layers until it gets real arrays.
    """
    import numpy as np
    import h5py

    if "records" not in f:
        raise ValueError("master_lib.mat must contain 'records' (v7.3).")

    def deref(x):
        """
        Dereference:
          - h5py.Reference -> target object
          - dataset with references -> follow reference if scalar
          - dataset with numeric -> return np.array
        """
        # Direct reference
        if isinstance(x, h5py.Reference):
            return f[x]

        # h5py Dataset or Group
        if hasattr(x, "shape") and hasattr(x, "__getitem__"):
            # Dataset
            if isinstance(x, h5py.Dataset):
                arr = x[()]
                # If dataset stores references
                if isinstance(arr, h5py.Reference):
                    return f[arr]
                # If array of references, return as-is (caller handles)
                return arr
            # Group: return it
            return x

        return x

    def read_numeric(obj) -> np.ndarray:
        """
        Read numeric data from a v7.3 MATLAB HDF5 object, following references until
        we get an actual numeric ndarray (ideally Nx2 for contours).
        """
        import numpy as np
        import h5py

        # Unwrap references
        obj = deref(obj)

        # If it's a dataset, get its payload
        if isinstance(obj, h5py.Dataset):
            payload = obj[()]
            # Dataset payload can itself be a reference (scalar) or array of refs
            return read_numeric(payload)

        # If it's a group, try common conventions:
        # - some MATLAB exports store actual numeric array under a child dataset
        if hasattr(obj, "keys"):
            # Prefer a dataset child if exists
            for k in obj.keys():
                child = obj[k]
                if isinstance(child, h5py.Dataset):
                    return read_numeric(child)
            raise ValueError("Group did not contain any datasets to read numeric data from.")

        # Now obj should be either:
        # - h5py.Reference
        # - numpy array
        # - scalar numeric
        if isinstance(obj, h5py.Reference):
            return read_numeric(f[obj])

        arr = np.array(obj)

        # Key fix: scalar array containing a reference
        if arr.shape == () and isinstance(arr.item(), h5py.Reference):
            return read_numeric(arr.item())

        # Also handle arrays of references (object arrays or dtype=ref)
        if arr.dtype == object:
            # If it’s a 1-element object array containing a ref, unwrap it
            flat = arr.ravel()
            if flat.size == 1 and isinstance(flat[0], h5py.Reference):
                return read_numeric(flat[0])
            # Otherwise this is a more complex layout
            raise ValueError("Numeric read got dtype=object array. Need a small layout-specific adaptation.")

        # Some h5py reference arrays have special dtype kind
        # If entries are references, unwrap singletons
        if arr.shape == () and hasattr(arr, "dtype") and "ref" in str(arr.dtype).lower():
            try:
                return read_numeric(arr.item())
            except Exception:
                pass

        return arr


    def read_matlab_string(obj) -> str:
        import numpy as np
        import h5py

        obj = deref(obj)

        # Dataset -> payload
        if isinstance(obj, h5py.Dataset):
            payload = obj[()]
            return read_matlab_string(payload)

        # Direct reference -> follow
        if isinstance(obj, h5py.Reference):
            return read_matlab_string(f[obj])

        arr = np.array(obj)
        arr = np.squeeze(arr)

        # Scalar array containing a reference
        if arr.shape == () and isinstance(arr.item(), h5py.Reference):
            return read_matlab_string(arr.item())

        # If it's an array of references (e.g., char stored indirectly)
        if arr.dtype == object:
            flat = arr.ravel()
            if flat.size == 1 and isinstance(flat[0], h5py.Reference):
                return read_matlab_string(flat[0])

        # Bytes / unicode
        if arr.dtype.kind in {"S", "U"}:
            return str(arr)

        # uint16 char codes
        if arr.dtype.kind in {"u", "i"}:
            try:
                return "".join(chr(int(c)) for c in arr.flatten() if int(c) != 0).strip()
            except Exception:
                return ""

        return ""


    records_obj = f["records"]

    # Case A: records is a dataset of object references (most common)
    if isinstance(records_obj, h5py.Dataset):
        refs = records_obj[()]
        refs = np.array(refs).ravel()

        out: List[Record] = []
        for ref in refs:
            if not isinstance(ref, h5py.Reference):
                continue
            rec = f[ref]  # group-like struct

            # category
            cat = ""
            if "category" in rec:
                cat = read_matlab_string(rec["category"])

            # contour
            if "contour" not in rec:
                continue
            contour_raw = deref(rec["contour"])
            # contour itself may be a reference
            if isinstance(contour_raw, h5py.Reference):
                contour_raw = f[contour_raw]
            contour_u = _as_2col_array(np.array(read_numeric(contour_raw)))

            # img_id optional
            img_id = None
            if "img_id" in rec:
                try:
                    img_id_arr = np.array(read_numeric(rec["img_id"]))
                    img_id = int(np.squeeze(img_id_arr))
                except Exception:
                    img_id = None

            out.append(Record(category=cat, contour_u=contour_u, img_id=img_id))

        if not out:
            raise ValueError("No records extracted from v7.3 'records' references.")
        return out

    # Case B: records is a group with fields (rarer)
    if hasattr(records_obj, "keys"):
        # If it's stored as separate field datasets, this is a different layout.
        # We fall back to your previous “records/category + records/contour” assumption,
        # but with dereferencing.
        if "category" in records_obj and "contour" in records_obj:
            cats = records_obj["category"]
            conts = records_obj["contour"]
            img_ids = records_obj["img_id"] if "img_id" in records_obj else None

            n = np.array(cats).shape[0]
            out: List[Record] = []
            for i in range(n):
                cat = read_matlab_string(deref(cats[i]))
                contour_u = _as_2col_array(np.array(read_numeric(deref(conts[i]))))
                img_id = None
                if img_ids is not None:
                    try:
                        img_id = int(np.squeeze(np.array(read_numeric(deref(img_ids[i])))))
                    except Exception:
                        img_id = None
                out.append(Record(category=cat, contour_u=contour_u, img_id=img_id))

            if not out:
                raise ValueError("No records extracted from v7.3 grouped layout.")
            return out

    raise ValueError("Unsupported v7.3 'records' layout. Need minor adaptation.")



def load_master_records(mat_path: str | Path) -> List[Record]:
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Not found: {mat_path}")

    d = _try_load_classic_mat(mat_path)
    if d is not None:
        return _extract_records_classic(d)

    f = _try_load_v73_mat(mat_path)
    if f is not None:
        try:
            return _extract_records_v73(f)
        finally:
            f.close()

    raise RuntimeError("Could not read master_lib.mat. Install scipy (classic) or h5py (v7.3).")


def build_class_index(records: List[Record]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Returns:
      classes: sorted unique categories
      byClass: dict category -> numpy array of indices into records
    """
    cats = [r.category for r in records]
    classes = sorted({c for c in cats if c is not None and c != ""})

    byClass: Dict[str, np.ndarray] = {}
    for c in classes:
        idx = np.where(np.array(cats, dtype=object) == c)[0]
        byClass[c] = idx

    return classes, byClass
