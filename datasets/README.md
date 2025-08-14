# BDD100K Dataset Download Instructions

## 📥 Required Files

You need to download the following files and place them in this `datasets/` directory:

1. **bdd100k_images_100k.zip** (5.3GB) - Real autonomous driving images
2. **bdd100k_labels.zip** (181MB) - Object detection labels

## 🔗 Download Link

**Official BDD100K Download Portal**: [http://bdd-data.berkeley.edu/download.html](http://bdd-data.berkeley.edu/download.html)

## 📋 Steps to Download

1. Visit [http://bdd-data.berkeley.edu/download.html](http://bdd-data.berkeley.edu/download.html)
2. Download the following files directly (no registration required):
   - `bdd100k_images_100k.zip` (5.3GB)
   - `bdd100k_labels.zip` (181MB)
3. Place both files in this `datasets/` directory

## 📁 Final Directory Structure

After downloading, your `datasets/` folder should look like this:

```
datasets/
├── README.md                  # This file
├── bdd100k_loader.py         # Data loader (already included)
├── bdd100k_images_100k.zip   # 5.3GB - Downloaded file
└── bdd100k_labels.zip        # 181MB - Downloaded file
```

## ✅ Verification

To verify the files are downloaded correctly:

```bash
# Check file sizes
du -h datasets/*.zip

# Expected output:
# 5.3G datasets/bdd100k_images_100k.zip
# 181M datasets/bdd100k_labels.zip
```

## ⚠️ Important Notes

- **Direct Download**: No registration required - download directly from the provided link
- **File Size**: Total download is ~5.5GB, ensure you have sufficient bandwidth and storage
- **Academic Use**: The dataset is provided for academic research purposes
- **Demo Ready**: Once downloaded, run `python3 complete_from_scratch_demo.py` from the project root

## 🆘 Troubleshooting

**If download fails:**
- Check your internet connection
- Try downloading during off-peak hours
- Use a download manager for large files
- Contact BDD100K support if issues persist

**If files are corrupted:**
- Re-download the files
- Verify file sizes match the expected values above
- Check available disk space before downloading
