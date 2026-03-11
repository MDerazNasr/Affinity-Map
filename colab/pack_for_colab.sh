#!/usr/bin/env bash
# pack_for_colab.sh
# Run from the project root: bash colab/pack_for_colab.sh
# Creates protein_fewshot.zip — upload this to Google Drive root.

set -e
cd "$(dirname "$0")/.."

ZIP="protein_fewshot.zip"
rm -f "$ZIP"

zip -r "$ZIP" \
  data/processed/proteins.json \
  utils/ \
  models/ \
  colab/ \
  -x "*.pyc" \
  -x "*/__pycache__/*" \
  -x "*.DS_Store"

SIZE=$(du -sh "$ZIP" | cut -f1)
echo ""
echo "Created $ZIP ($SIZE)"
echo ""
echo "Next steps:"
echo "  1. Upload $ZIP to Google Drive root (MyDrive/)"
echo "  2. Open colab/run_lora.ipynb in Google Colab"
echo "  3. Runtime → Change runtime type → GPU"
echo "  4. Run all cells top to bottom"
echo ""
echo "Expected runtime: ~1.5 hrs on T4 free, ~45 min on A100 (Colab Pro)"
