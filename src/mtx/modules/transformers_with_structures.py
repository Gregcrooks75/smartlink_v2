"""Historical import path for the transformer/structure workbook loader."""
import sys

from mtx.data import transformers_with_structures as implementation

if __name__ == "__main__":
    print(implementation.merge_transformer_and_structure().head())
else:
    sys.modules[__name__] = implementation
