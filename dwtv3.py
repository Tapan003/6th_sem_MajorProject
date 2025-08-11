import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# === Load Images ===
secret_img = cv2.imread('secret.png', cv2.IMREAD_GRAYSCALE)
if secret_img is None:
    raise ValueError("Secret image not found.")

cover_img = cv2.imread('cover.png', cv2.IMREAD_GRAYSCALE)
if cover_img is None:
    raise ValueError("Cover image not found.")

# === Resize ===
secret_img = cv2.resize(secret_img, (256, 256))
cover_img = cv2.resize(cover_img, (256, 256))

# === DWT Transformations ===
secret_float = np.float32(secret_img)
cover_float = np.float32(cover_img)

# Apply 2D DWT using Daubechies wavelet
wavelet = 'db4'  # Daubechies 4 wavelet

# Extract subbands: (LL, (LH, HL, HH))
secret_LL, (secret_LH, secret_HL, secret_HH) = pywt.dwt2(secret_float, wavelet)
cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_float, wavelet)

# Create visualization of secret DWT coefficients
# secret_dwt_vis = np.abs(secret_LL)  # Use LL subband for visualization
# secret_dwt_log = np.log(secret_dwt_vis + 1)

# === EMBEDDING STRATEGY ===
# Hide secret_LL (most important) in cover_HH (high frequency - less perceptible)
# Hide secret_LH, secret_HL in their corresponding cover subbands
# This way we use all secret information but prioritize hiding LL in high freq
alpha = 0.05

embedded_LL = cover_LL.copy()  # Keep cover LL unchanged
embedded_LH = cover_LH + alpha * secret_LH  # Hide secret LH details
embedded_HL = cover_HL + alpha * secret_HL  # Hide secret HL details  
embedded_HH = cover_HH + alpha * secret_LL  # Hide secret LL (main component) in HH

# print("Embedding Strategy:")
# print("- Secret LL (main component) → Cover HH (high frequency)")
# print("- Secret LH (horizontal details) → Cover LH") 
# print("- Secret HL (vertical details) → Cover HL")
# print("- Cover LL unchanged (preserves overall appearance)")

# Reconstruct embedded coefficients
embedded_coeffs = (embedded_LL, (embedded_LH, embedded_HL, embedded_HH))

# IDWT: Get Watermarked Image
watermarked_img = pywt.idwt2(embedded_coeffs, wavelet)

# === EXTRACTION PROCESS (CORRECTED) ===
watermarked_coeffs = pywt.dwt2(watermarked_img, wavelet)
watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH) = watermarked_coeffs

# Extract secret coefficients - MATCH the embedding strategy!
extracted_secret_LL = (watermarked_HH - cover_HH) / alpha  # LL was hidden in HH!
extracted_secret_LH = (watermarked_LH - cover_LH) / alpha  # LH was hidden in LH
extracted_secret_HL = (watermarked_HL - cover_HL) / alpha  # HL was hidden in HL
extracted_secret_HH = np.zeros_like(secret_HH)  # We didn't embed original HH, so set to zero

# print("\nExtraction Strategy:")
# print("- Extract secret LL from watermarked HH")
# print("- Extract secret LH from watermarked LH") 
# print("- Extract secret HL from watermarked HL")
# print("- Secret HH set to zero (wasn't embedded)")

# Reconstruct extracted secret
extracted_secret_coeffs = (extracted_secret_LL, (extracted_secret_LH, extracted_secret_HL, extracted_secret_HH))
extracted_secret = pywt.idwt2(extracted_secret_coeffs, wavelet)

# === ALTERNATIVE: Use Original HH for Better Quality ===
# If you want even better reconstruction, you could store secret_HH separately
# or use a different embedding strategy for HH
# extracted_secret_HH_alt = secret_HH.copy()  # Use original HH
# extracted_secret_coeffs_alt = (extracted_secret_LL, (extracted_secret_LH, extracted_secret_HL, extracted_secret_HH_alt))
# extracted_secret_alt = pywt.idwt2(extracted_secret_coeffs_alt, wavelet)

# === Display and Save Output ===
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

axs[0, 0].imshow(cover_img, cmap='gray')
axs[0, 0].set_title("Original Cover Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(secret_img, cmap='gray')
axs[0, 1].set_title("Original Secret Image")
axs[0, 1].axis('off')

# axs[0, 2].imshow(secret_dwt_log, cmap='gray')
# axs[0, 2].set_title("Secret DWT LL Coefficients (Log Scale)")
axs[0, 2].axis('off')

axs[1, 0].imshow(np.clip(watermarked_img, 0, 255), cmap='gray')
axs[1, 0].set_title("Watermarked Image (After IDWT)")
axs[1, 0].axis('off')

axs[1, 1].imshow(np.clip(extracted_secret, 0, 255), cmap='gray')
axs[1, 1].set_title("Extracted Secret Image")
axs[1, 1].axis('off')

# Show the difference between original and extracted secret
# difference = np.abs(secret_img - np.clip(extracted_secret, 0, 255))
# axs[1, 2].imshow(difference, cmap='hot')
# axs[1, 2].set_title("Difference (Original - Extracted)")
axs[1, 2].axis('off')

plt.tight_layout()
# plt.savefig('output_result_dwt_corrected.png')
plt.show()

# # === Statistics ===
# print(f"\n=== STATISTICS ===")
# print(f"Original secret image shape: {secret_img.shape}")
# print(f"Watermarked image range: [{np.min(watermarked_img):.2f}, {np.max(watermarked_img):.2f}]")
# print(f"Extracted secret image range: [{np.min(extracted_secret):.2f}, {np.max(extracted_secret):.2f}]")
# print(f"MSE between original and extracted secret: {np.mean((secret_img - np.clip(extracted_secret, 0, 255))**2):.2f}")

# # Calculate PSNR for quality assessment
# def calculate_psnr(img1, img2):
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 20 * np.log10(255.0 / np.sqrt(mse))

# psnr_watermarked = calculate_psnr(cover_float, np.clip(watermarked_img, 0, 255))
# psnr_extracted = calculate_psnr(secret_img, np.clip(extracted_secret, 0, 255))

# print(f"PSNR (Cover vs Watermarked): {psnr_watermarked:.2f} dB")
# print(f"PSNR (Original vs Extracted Secret): {psnr_extracted:.2f} dB")

# # Show what's embedded where
# print(f"\n=== EMBEDDING ANALYSIS ===")
# print(f"Secret LL energy (embedded in HH): {np.sum(secret_LL**2):.0f}")
# print(f"Secret LH energy: {np.sum(secret_LH**2):.0f}")
# print(f"Secret HL energy: {np.sum(secret_HL**2):.0f}")
# print(f"Secret HH energy (lost): {np.sum(secret_HH**2):.0f}")
# print(f"Total energy preserved: {(np.sum(secret_LL**2) + np.sum(secret_LH**2) + np.sum(secret_HL**2)) / np.sum(secret_img**2) * 100:.1f}%")