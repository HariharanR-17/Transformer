import kagglehub
kagglehub.login()
LOCAL_MODEL_DIR = '/kaggle/working/robertalargeuf_best.pth'
MODEL_SLUG = 'RoBERTaL'
VARIATION_SLUG = 'Finetuned'
kagglehub.model_upload(
  handle = f"hariharanr2003/{MODEL_SLUG}/pyTorch/{VARIATION_SLUG}",
  local_model_dir = LOCAL_MODEL_DIR,
  version_notes = 'Update 2025-11-29')
