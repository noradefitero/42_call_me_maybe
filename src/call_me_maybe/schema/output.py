from pydantic import RootModel

from .fn_calling import FnCalling

Output = RootModel[list[FnCalling]]
