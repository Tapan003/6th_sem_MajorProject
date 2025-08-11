# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pywt

# # === Load Images ===
# secret_img = cv2.imread('secret2.png', cv2.IMREAD_GRAYSCALE)
# if secret_img is None:
#     raise ValueError("Secret image not found.")

# cover_img = cv2.imread('cover.png', cv2.IMREAD_GRAYSCALE)
# if cover_img is None:
#     raise ValueError("Cover image not found.")

# # === Resize ===
# secret_img = cv2.resize(secret_img, (256, 256))
# cover_img = cv2.resize(cover_img, (256, 256))

# # === DWT Transformations ===
# secret_float = np.float32(secret_img)
# cover_float = np.float32(cover_img)

# # Apply 2D DWT using Daubechies wavelet
# wavelet = 'db4'  # Daubechies 4 wavelet
# # secret_coeffs = pywt.dwt2(secret_float, wavelet)
# # cover_coeffs = pywt.dwt2(cover_float, wavelet)

# # Extract subbands: (LL, (LH, HL, HH))
# # secret_LL, (secret_LH, secret_HL, secret_HH) = secret_coeffs
# # cover_LL, (cover_LH, cover_HL, cover_HH) = cover_coeffs

# secret_LL, (secret_LH, secret_HL, secret_HH) = pywt.dwt2(secret_float, wavelet)
# cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_float, wavelet)

# # Create visualization of secret DWT coefficients (similar to DCT log scale)
# # secret_dwt_vis = np.abs(secret_LL)  # Use LL subband for visualization
# # secret_dwt_log = np.log(secret_dwt_vis + 1)

# # Embed Secret into Cover using DWT coefficients
# # alpha = 0.05

# # Method 1: Embed in all subbands
# # embedded_LL = cover_LL + alpha * secret_LL
# # embedded_LH = cover_LH + alpha * secret_LH
# # embedded_HL = cover_HL + alpha * secret_HL
# # embedded_HH = cover_HH + alpha * secret_HH

# # Alternative Method 2: Embed only in mid-frequency subbands (LH, HL, HH)
# # This is more similar to the DCT mid-frequency embedding approach
# # embedded_LL = cover_LL.copy()  # Keep LL unchanged
# # embedded_LH = cover_LH + alpha * secret_LH
# # embedded_HL = cover_HL + alpha * secret_HL
# # embedded_HH = cover_HH + alpha * secret_HH

# embedded_LL = cover_LL.copy()  # Keep LL unchanged
# embedded_LH = cover_LH.copy()
# embedded_HL = cover_HL.copy()
# embedded_HH = secret_LL.copy()

# # Reconstruct embedded coefficients
# embedded_coeffs = (embedded_LL, (embedded_LH, embedded_HL, embedded_HH))

# # IDWT: Get Watermarked Image
# watermarked_img = pywt.idwt2(embedded_coeffs, wavelet)

# # Extraction Process
# watermarked_coeffs = pywt.dwt2(watermarked_img, wavelet)
# watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH) = watermarked_coeffs

# # Extract secret coefficients
# extracted_secret_LL = (watermarked_LL - cover_LL)
# extracted_secret_LH = (watermarked_LH - cover_LH)
# extracted_secret_HL = (watermarked_HL - cover_HL)
# extracted_secret_HH = (watermarked_HH - secret_LL)

# # Reconstruct extracted secret
# extracted_secret_coeffs = (extracted_secret_LL, (extracted_secret_LH, extracted_secret_HL, extracted_secret_HH))
# extracted_secret = pywt.idwt2(extracted_secret_coeffs, wavelet)

# # === Display and Save Output ===
# fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# axs[0, 0].imshow(cover_img, cmap='gray')
# axs[0, 0].set_title("Original Cover Image")
# axs[0, 0].axis('off')

# axs[0, 1].imshow(secret_img, cmap='gray')
# axs[0, 1].set_title("Original Secret Image")
# axs[0, 1].axis('off')

# # axs[0, 2].imshow(secret_dwt_log, cmap='gray')
# # axs[0, 2].set_title("Secret DWT LL Coefficients (Log Scale)")
# axs[0, 2].axis('off')

# axs[1, 0].imshow(np.clip(watermarked_img, 0, 255), cmap='gray')
# axs[1, 0].set_title("Watermarked Image (After IDWT)")
# axs[1, 0].axis('off')

# axs[1, 1].imshow(np.clip(extracted_secret, 0, 255), cmap='gray')
# axs[1, 1].set_title("Extracted Secret Image")
# axs[1, 1].axis('off')

# axs[1, 2].axis('off')  # Empty panel

# plt.tight_layout()
# plt.savefig('output_result_dwt.png')  # Save output for inspection
# plt.show()

# # Optional: Print some statistics for comparison
# # print(f"Original secret image shape: {secret_img.shape}")
# # print(f"Watermarked image range: [{np.min(watermarked_img):.2f}, {np.max(watermarked_img):.2f}]")
# # print(f"Extracted secret image range: [{np.min(extracted_secret):.2f}, {np.max(extracted_secret):.2f}]")
# # print(f"MSE between original and extracted secret: {np.mean((secret_img - np.clip(extracted_secret, 0, 255))**2):.2f}")

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pywt

# # === Load Images ===
# secret_img = cv2.imread('secret.png', cv2.IMREAD_GRAYSCALE)
# if secret_img is None:
#     raise ValueError("Secret image not found.")

# cover_img = cv2.imread('cover.png', cv2.IMREAD_GRAYSCALE)
# if cover_img is None:
#     raise ValueError("Cover image not found.")

# # === Resize ===
# secret_img = cv2.resize(secret_img, (256, 256))
# cover_img = cv2.resize(cover_img, (256, 256))

# # === DWT Transformations ===
# secret_float = np.float32(secret_img)
# cover_float = np.float32(cover_img)

# # Apply 2D DWT using Daubechies wavelet
# wavelet = 'db8'  # Daubechies 4 wavelet

# # Extract subbands: (LL, (LH, HL, HH))
# secret_LL, (secret_LH, secret_HL, secret_HH) = pywt.dwt2(secret_float, wavelet)
# cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_float, wavelet)

# # print("=== EMBEDDING STRATEGY (Direct Replacement) ===")
# # print("- Cover LL: Keep unchanged")
# # print("- Cover LH: Keep unchanged") 
# # print("- Cover HL: Keep unchanged")
# # print("- Cover HH: REPLACE with Secret LL")

# # === DIRECT REPLACEMENT EMBEDDING ===
# embedded_LL = cover_LL     # Keep cover LL unchanged
# embedded_LH = cover_LH    # Keep cover LH unchanged
# embedded_HL = secret_LL/10    # Keep cover HL unchanged
# embedded_HH = secret_LH/10   # REPLACE cover HH with secret LL

# # Reconstruct embedded coefficients
# embedded_coeffs = (embedded_LL, (embedded_LH, embedded_HL, embedded_HH))

# # IDWT: Get Watermarked Image
# watermarked_img = pywt.idwt2(embedded_coeffs, wavelet)

# # === EXTRACTION PROCESS (CORRECTED) ===
# watermarked_coeffs = pywt.dwt2(watermarked_img, wavelet)
# watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH) = watermarked_coeffs

# # print("\n=== EXTRACTION STRATEGY (Corrected) ===")
# # print("- Extract Secret LL from watermarked HH (where we put it)")
# # print("- Other subbands: Set to zero or reconstruct differently")

# # CORRECTED EXTRACTION:
# # Since we directly replaced HH with secret_LL, we extract it directly
# extracted_secret_LL = watermarked_HL.copy() *10 # Secret LL is IN the HH position!

# # For other subbands - since we didn't embed them, we have options:
# # Option 1: Set to zero (what you had, but corrected)
# extracted_secret_LH = np.zeros_like(secret_LH)
# extracted_secret_HL = np.zeros_like(secret_HL) 
# extracted_secret_HH = np.zeros_like(secret_HH)

# # Option 2: Try to extract from unchanged subbands (will be mostly zeros)
# # extracted_secret_LH = watermarked_LH - cover_LH  # Should be ~zero
# # extracted_secret_HL = watermarked_HL - cover_HL  # Should be ~zero
# # extracted_secret_HH = watermarked_LL - cover_LL  # Should be ~zero

# # print(f"- Secret LH, HL, HH: Set to zero (not embedded)")

# # Reconstruct extracted secret
# extracted_secret_coeffs = (extracted_secret_LL, (extracted_secret_LH, extracted_secret_HL, extracted_secret_HH))
# extracted_secret = pywt.idwt2(extracted_secret_coeffs, wavelet)

# # === Display and Save Output ===
# fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# axs[0, 0].imshow(cover_img, cmap='gray')
# axs[0, 0].set_title("Original Cover Image")
# axs[0, 0].axis('off')

# axs[0, 1].imshow(secret_img, cmap='gray')
# axs[0, 1].set_title("Original Secret Image")
# axs[0, 1].axis('off')

# # Show what we embedded (secret LL)
# # secret_dwt_vis = np.abs(secret_LL)
# # secret_dwt_log = np.log(secret_dwt_vis + 1)
# # axs[0, 2].imshow(secret_dwt_log, cmap='gray')
# # axs[0, 2].set_title("Secret LL (What We Embedded)")
# axs[0, 2].axis('off')

# axs[1, 0].imshow(np.clip(watermarked_img, 0, 255), cmap='gray')
# axs[1, 0].set_title("Watermarked Image (After IDWT)")
# axs[1, 0].axis('off')

# axs[1, 1].imshow(np.clip(extracted_secret, 0, 255), cmap='gray')
# axs[1, 1].set_title("Extracted Secret Image")
# axs[1, 1].axis('off')

# # Show the extracted LL component
# # extracted_ll_vis = np.log(np.abs(extracted_secret_LL) + 1)
# # axs[1, 2].imshow(extracted_ll_vis, cmap='gray')
# # axs[1, 2].set_title("Extracted Secret LL")
# axs[1, 2].axis('off')

# plt.tight_layout()
# # plt.savefig('output_result_dwt_direct_replacement.png')
# plt.show()

# # === Analysis and Statistics ===
# print(f"\n=== STATISTICS ===")
# print(f"Original secret image shape: {secret_img.shape}")
# print(f"Watermarked image range: [{np.min(watermarked_img):.2f}, {np.max(watermarked_img):.2f}]")
# print(f"Extracted secret image range: [{np.min(extracted_secret):.2f}, {np.max(extracted_secret):.2f}]")
# print(f"MSE between original and extracted secret: {np.mean((secret_img - np.clip(extracted_secret, 0, 255))**2):.2f}")

# print(f"\n=== ENERGY ANALYSIS ===")
# print(f"Original Secret LL energy: {np.sum(secret_LL**2):.0f}")
# print(f"Extracted Secret LL energy: {np.sum(extracted_secret_LL**2):.0f}")
# print(f"LL preservation: {np.sum(extracted_secret_LL**2)/np.sum(secret_LL**2)*100:.1f}%")

# print(f"\n=== WHAT'S HAPPENING ===")
# print("1. We're taking the SECRET'S main component (LL)")
# print("2. Directly replacing the COVER'S high-frequency component (HH)")
# print("3. This preserves the cover's overall appearance (LL, LH, HL unchanged)")
# print("4. But we lose most secret detail information (LH, HL, HH)")
# print("5. Result: Extracted secret looks like a low-resolution version")

# # Verify our extraction is correct
# print(f"\n=== VERIFICATION ===")
# print(f"Secret LL and Extracted LL are identical: {np.allclose(secret_LL, extracted_secret_LL)}")
# print(f"Max difference between secret LL and extracted LL: {np.max(np.abs(secret_LL - extracted_secret_LL)):.6f}")

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
wavelet = 'db8'  # Daubechies 8 wavelet

# Extract subbands: (LL, (LH, HL, HH))
secret_LL, (secret_LH, secret_HL, secret_HH) = pywt.dwt2(secret_float, wavelet)
cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_float, wavelet)

print("=== EMBEDDING STRATEGY ===")
print("- Cover LL: Keep unchanged")
print("- Cover LH: Keep unchanged") 
print("- Cover HL: REPLACE with Secret LL / 10")
print("- Cover HH: REPLACE with Secret LH / 10")

# === CORRECTED EMBEDDING ===
embedded_LL = cover_LL.copy()        # Keep cover LL unchanged
embedded_LH = cover_LH.copy()        # Keep cover LH unchanged
embedded_HL = secret_LL / 10         # REPLACE cover HL with secret LL
embedded_HH = secret_LH / 10         # REPLACE cover HH with secret LH

# Reconstruct embedded coefficients
embedded_coeffs = (embedded_LL, (embedded_LH, embedded_HL, embedded_HH))

# IDWT: Get Watermarked Image
watermarked_img = pywt.idwt2(embedded_coeffs, wavelet)

# === CORRECTED EXTRACTION PROCESS ===
watermarked_coeffs = pywt.dwt2(watermarked_img, wavelet)
watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH) = watermarked_coeffs

print("\n=== CORRECTED EXTRACTION ===")
print("- Extract Secret LL from watermarked HL * 10")
print("- Extract Secret LH from watermarked HH * 10")
print("- Other subbands set to zero")

# CORRECTED EXTRACTION:
extracted_secret_LL = watermarked_HL * 10    # Extract secret LL from HL
extracted_secret_LH = watermarked_HH * 10    # Extract secret LH from HH
extracted_secret_HL = np.zeros_like(secret_HL)  # Not embedded, set to zero
extracted_secret_HH = np.zeros_like(secret_HH)  # Not embedded, set to zero

# Reconstruct extracted secret
extracted_secret_coeffs = (extracted_secret_LL, (extracted_secret_LH, extracted_secret_HL, extracted_secret_HH))
extracted_secret = pywt.idwt2(extracted_secret_coeffs, wavelet)

# === VERIFICATION ===
print(f"\n=== VERIFICATION ===")
print(f"Secret LL vs Extracted LL - Max difference: {np.max(np.abs(secret_LL - extracted_secret_LL)):.6f}")
print(f"Secret LH vs Extracted LH - Max difference: {np.max(np.abs(secret_LH - extracted_secret_LH)):.6f}")

# === Display and Save Output ===
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

axs[0, 0].imshow(cover_img, cmap='gray')
axs[0, 0].set_title("Original Cover Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(secret_img, cmap='gray')
axs[0, 1].set_title("Original Secret Image")
axs[0, 1].axis('off')

# Show what we're embedding
secret_ll_vis = np.log(np.abs(secret_LL) + 1)
axs[0, 2].imshow(secret_ll_vis, cmap='gray')
axs[0, 2].set_title("Secret LL (→HL)")
axs[0, 2].axis('off')

secret_lh_vis = np.log(np.abs(secret_LH) + 1)
axs[0, 3].imshow(secret_lh_vis, cmap='gray')
axs[0, 3].set_title("Secret LH (→HH)")
axs[0, 3].axis('off')

axs[1, 0].imshow(np.clip(watermarked_img, 0, 255), cmap='gray')
axs[1, 0].set_title("Watermarked Image")
axs[1, 0].axis('off')

axs[1, 1].imshow(np.clip(extracted_secret, 0, 255), cmap='gray')
axs[1, 1].set_title("Extracted Secret Image")
axs[1, 1].axis('off')

# Show extracted components
extracted_ll_vis = np.log(np.abs(extracted_secret_LL) + 1)
axs[1, 2].imshow(extracted_ll_vis, cmap='gray')
axs[1, 2].set_title("Extracted LL")
axs[1, 2].axis('off')

extracted_lh_vis = np.log(np.abs(extracted_secret_LH) + 1)
axs[1, 3].imshow(extracted_lh_vis, cmap='gray')
axs[1, 3].set_title("Extracted LH")
axs[1, 3].axis('off')

plt.tight_layout()
plt.savefig('corrected_dwt_steganography.png', dpi=150, bbox_inches='tight')
plt.show()

# === STATISTICS ===
print(f"\n=== STATISTICS ===")
print(f"Original secret image range: [{np.min(secret_img)}, {np.max(secret_img)}]")
print(f"Watermarked image range: [{np.min(watermarked_img):.2f}, {np.max(watermarked_img):.2f}]")
print(f"Extracted secret range: [{np.min(extracted_secret):.2f}, {np.max(extracted_secret):.2f}]")

# Calculate reconstruction quality
mse = np.mean((secret_img - np.clip(extracted_secret, 0, 255))**2)
print(f"MSE between original and extracted secret: {mse:.2f}")

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

psnr_watermarked = calculate_psnr(cover_float, np.clip(watermarked_img, 0, 255))
psnr_extracted = calculate_psnr(secret_img, np.clip(extracted_secret, 0, 255))

print(f"PSNR (Cover vs Watermarked): {psnr_watermarked:.2f} dB")
print(f"PSNR (Original vs Extracted): {psnr_extracted:.2f} dB")

print(f"\n=== POTENTIAL ISSUES TO WATCH ===")
print("1. Division by 10 reduces signal strength - may cause information loss")
print("2. Only using LL and LH - losing HL and HH detail information")
print("3. Range issues: Make sure extracted values don't exceed [0,255]")
print("4. Statistical mismatch: HL/HH subbands have different characteristics than LL/LH")

# === ADDITIONAL CHECKS ===
print(f"\n=== RANGE CHECKS ===")
print(f"Embedded HL range: [{np.min(embedded_HL):.2f}, {np.max(embedded_HL):.2f}]")
print(f"Embedded HH range: [{np.min(embedded_HH):.2f}, {np.max(embedded_HH):.2f}]")
print(f"Original cover HL range: [{np.min(cover_HL):.2f}, {np.max(cover_HL):.2f}]")
print(f"Original cover HH range: [{np.min(cover_HH):.2f}, {np.max(cover_HH):.2f}]")

# Check if the embedded values are reasonable
if np.max(np.abs(embedded_HL - cover_HL)) > 50:
    print("⚠️  WARNING: Large difference in HL subband - may cause visible artifacts")
if np.max(np.abs(embedded_HH - cover_HH)) > 50:
    print("⚠️  WARNING: Large difference in HH subband - may cause visible artifacts")

if np.max(extracted_secret) > 255 or np.min(extracted_secret) < 0:
    print("⚠️  WARNING: Extracted secret values outside [0,255] range")
    
print(f"\n=== INFORMATION PRESERVATION ===")
original_energy = np.sum(secret_img**2)
ll_energy = np.sum(secret_LL**2)
lh_energy = np.sum(secret_LH**2)
hl_energy = np.sum(secret_HL**2)
hh_energy = np.sum(secret_HH**2)

print(f"Energy preserved: {(ll_energy + lh_energy)/original_energy*100:.1f}%")
print(f"Energy lost (HL+HH): {(hl_energy + hh_energy)/original_energy*100:.1f}%")