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

# === ADAPTIVE SCALING FUNCTIONS ===
def adaptive_scaling_embed(secret_subband, cover_subband, method='statistical'):
    """
    Adaptive scaling for embedding to prevent overflow/underflow
    
    Methods:
    - 'statistical': Match mean and std of cover subband
    - 'range': Scale to fit within cover subband range
    - 'energy': Preserve relative energy while fitting range
    """
    
    if method == 'statistical':
        # Method 1: Statistical Matching (Best for blind extraction)
        secret_mean = np.mean(secret_subband)
        secret_std = np.std(secret_subband)
        cover_mean = np.mean(cover_subband)
        cover_std = np.std(cover_subband)
        
        if secret_std > 0:
            # Normalize secret to zero mean, unit variance, then scale to cover stats
            normalized = (secret_subband - secret_mean) / secret_std
            scaled = normalized * (cover_std * 0.1) + cover_mean  # 0.1 = strength factor
        else:
            scaled = np.full_like(secret_subband, cover_mean)
            
        scaling_params = {
            'secret_mean': secret_mean,
            'secret_std': secret_std,
            'cover_mean': cover_mean,
            'cover_std': cover_std,
            'strength': 0.1
        }
        
    elif method == 'range':
        # Method 2: Range Mapping
        secret_min, secret_max = np.min(secret_subband), np.max(secret_subband)
        cover_min, cover_max = np.min(cover_subband), np.max(cover_subband)
        
        if secret_max != secret_min:
            # Map secret range to a fraction of cover range
            cover_range = cover_max - cover_min
            target_range = cover_range * 0.2  # Use 20% of cover range
            
            # Normalize to [0,1] then scale to target range
            normalized = (secret_subband - secret_min) / (secret_max - secret_min)
            scaled = normalized * target_range + cover_min
        else:
            scaled = np.full_like(secret_subband, cover_min)
            
        scaling_params = {
            'secret_min': secret_min,
            'secret_max': secret_max,
            'cover_min': cover_min,
            'target_range': target_range
        }
        
    elif method == 'energy':
        # Method 3: Energy-based scaling
        secret_energy = np.sum(secret_subband**2)
        cover_energy = np.sum(cover_subband**2)
        
        if secret_energy > 0:
            # Scale to have 5% of cover energy
            energy_ratio = np.sqrt(cover_energy * 0.05 / secret_energy)
            scaled = secret_subband * energy_ratio
        else:
            scaled = secret_subband
            
        scaling_params = {
            'energy_ratio': energy_ratio if secret_energy > 0 else 1.0
        }
    
    return scaled, scaling_params

def adaptive_scaling_extract(embedded_subband, scaling_params, method='statistical'):
    """Extract and reverse the adaptive scaling"""
    
    if method == 'statistical':
        # Reverse statistical scaling
        cover_mean = scaling_params['cover_mean']
        cover_std = scaling_params['cover_std']
        secret_mean = scaling_params['secret_mean']
        secret_std = scaling_params['secret_std']
        strength = scaling_params['strength']
        
        # Reverse the embedding process
        normalized = (embedded_subband - cover_mean) / (cover_std * strength)
        extracted = normalized * secret_std + secret_mean
        
    elif method == 'range':
        # Reverse range mapping
        cover_min = scaling_params['cover_min']
        target_range = scaling_params['target_range']
        secret_min = scaling_params['secret_min']
        secret_max = scaling_params['secret_max']
        
        # Reverse normalization
        normalized = (embedded_subband - cover_min) / target_range
        extracted = normalized * (secret_max - secret_min) + secret_min
        
    elif method == 'energy':
        # Reverse energy scaling
        energy_ratio = scaling_params['energy_ratio']
        extracted = embedded_subband / energy_ratio
    
    return extracted

# === CHOOSE SCALING METHOD ===
scaling_method = 'energy'  # Options: 'statistical', 'range', 'energy'

print(f"=== USING {scaling_method.upper()} SCALING METHOD ===")

# === EMBEDDING WITH ADAPTIVE SCALING ===
embedded_LL = cover_LL.copy()
embedded_LH = cover_LH.copy()

# Scale secret coefficients adaptively
scaled_secret_LL, scaling_params_LL = adaptive_scaling_embed(secret_LL, cover_HL, scaling_method)
scaled_secret_LH, scaling_params_LH = adaptive_scaling_embed(secret_LH, cover_HH, scaling_method)

embedded_HL = scaled_secret_LL  # Embed scaled secret LL in HL
embedded_HH = scaled_secret_LH  # Embed scaled secret LH in HH

print(f"Scaling parameters stored for blind extraction:")
print(f"LL scaling: {list(scaling_params_LL.keys())}")
print(f"LH scaling: {list(scaling_params_LH.keys())}")

# Reconstruct embedded coefficients
embedded_coeffs = (embedded_LL, (embedded_LH, embedded_HL, embedded_HH))

# IDWT: Get Watermarked Image
watermarked_img = pywt.idwt2(embedded_coeffs, wavelet)

# === BLIND EXTRACTION (No Cover Image Needed) ===
print(f"\n=== BLIND EXTRACTION PROCESS ===")
watermarked_coeffs = pywt.dwt2(watermarked_img, wavelet)
watermarked_LL, (watermarked_LH, watermarked_HL, watermarked_HH) = watermarked_coeffs

# Extract scaled coefficients directly from watermarked image
extracted_scaled_LL = watermarked_HL  # Secret LL is embedded in HL
extracted_scaled_LH = watermarked_HH  # Secret LH is embedded in HH

# Reverse the scaling (this requires the scaling parameters)
# In practice, these parameters would be stored as metadata or derived
extracted_secret_LL = adaptive_scaling_extract(extracted_scaled_LL, scaling_params_LL, scaling_method)
extracted_secret_LH = adaptive_scaling_extract(extracted_scaled_LH, scaling_params_LH, scaling_method)

# Set non-embedded subbands to zero
extracted_secret_HL = np.zeros_like(secret_HL)
extracted_secret_HH = np.zeros_like(secret_HH)

# Reconstruct extracted secret
extracted_secret_coeffs = (extracted_secret_LL, (extracted_secret_LH, extracted_secret_HL, extracted_secret_HH))
extracted_secret = pywt.idwt2(extracted_secret_coeffs, wavelet)

# # === ALTERNATIVE: PARAMETER-FREE BLIND EXTRACTION ===
# def parameter_free_extraction(watermarked_img, expected_range=(0, 255)):
#     """
#     Extract without explicit scaling parameters by making assumptions
#     about the expected output range
#     """
#     watermarked_coeffs = pywt.dwt2(watermarked_img, wavelet)
#     wm_LL, (wm_LH, wm_HL, wm_HH) = watermarked_coeffs
    
#     # Assume embedded coefficients should produce values in expected_range
#     # Use statistical properties to reverse-engineer scaling
    
#     # Simple approach: normalize extracted subbands to expected range
#     def normalize_to_range(subband, target_min, target_max):
#         current_min, current_max = np.min(subband), np.max(subband)
#         if current_max != current_min:
#             normalized = (subband - current_min) / (current_max - current_min)
#             return normalized * (target_max - target_min) + target_min
#         return subband
    
#     # Extract and normalize
#     ext_LL = normalize_to_range(wm_HL, 0, 255)
#     ext_LH = normalize_to_range(wm_HH, -50, 50)  # Detail coefficients typically smaller
    
#     ext_coeffs = (ext_LL, (ext_LH, np.zeros_like(wm_HL), np.zeros_like(wm_HH)))
#     return pywt.idwt2(ext_coeffs, wavelet)

# # Test parameter-free extraction
# extracted_secret_alt = parameter_free_extraction(watermarked_img)

# === DISPLAY OUTPUT ===
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

axs[0, 0].imshow(cover_img, cmap='gray')
axs[0, 0].set_title("Original Cover Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(secret_img, cmap='gray')
axs[0, 1].set_title("Original Secret Image")
axs[0, 1].axis('off')

axs[0, 2].imshow(np.clip(watermarked_img, 0, 255), cmap='gray')
axs[0, 2].set_title("Watermarked Image")
axs[0, 2].axis('off')

axs[1, 0].imshow(np.clip(extracted_secret, 0, 255), cmap='gray')
axs[1, 0].set_title("Extracted Secret\n(With Parameters)")
axs[1, 0].axis('off')

# axs[1, 1].imshow(np.clip(extracted_secret_alt, 0, 255), cmap='gray')
# axs[1, 1].set_title("Extracted Secret\n(Parameter-Free)")
# axs[1, 1].axis('off')

# Show difference between methods
diff = np.abs(np.clip(extracted_secret, 0, 255) - np.clip(extracted_secret_alt, 0, 255))
axs[1, 2].imshow(diff, cmap='hot')
axs[1, 2].set_title("Difference Between Methods")
axs[1, 2].axis('off')

plt.tight_layout()
plt.savefig('blind_steganography_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# === STATISTICS AND ANALYSIS ===
print(f"\n=== QUALITY METRICS ===")

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

def calculate_ssim_simple(img1, img2):
    """Simplified SSIM calculation"""
    mu1, mu2 = np.mean(img1), np.mean(img2)
    var1, var2 = np.var(img1), np.var(img2)
    covar = np.mean((img1 - mu1) * (img2 - mu2))
    
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2*mu1*mu2 + c1) * (2*covar + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
    return ssim

# Quality metrics
psnr_watermarked = calculate_psnr(cover_float, np.clip(watermarked_img, 0, 255))
psnr_extracted_param = calculate_psnr(secret_img, np.clip(extracted_secret, 0, 255))
psnr_extracted_free = calculate_psnr(secret_img, np.clip(extracted_secret_alt, 0, 255))

ssim_extracted_param = calculate_ssim_simple(secret_img, np.clip(extracted_secret, 0, 255))
ssim_extracted_free = calculate_ssim_simple(secret_img, np.clip(extracted_secret_alt, 0, 255))

print(f"PSNR (Cover vs Watermarked): {psnr_watermarked:.2f} dB")
print(f"PSNR (Secret vs Extracted with params): {psnr_extracted_param:.2f} dB")
print(f"PSNR (Secret vs Extracted parameter-free): {psnr_extracted_free:.2f} dB")
print(f"SSIM (Secret vs Extracted with params): {ssim_extracted_param:.3f}")
print(f"SSIM (Secret vs Extracted parameter-free): {ssim_extracted_free:.3f}")

# Range checks
print(f"\n=== RANGE ANALYSIS ===")
print(f"Watermarked image range: [{np.min(watermarked_img):.2f}, {np.max(watermarked_img):.2f}]")
print(f"Extracted secret range (with params): [{np.min(extracted_secret):.2f}, {np.max(extracted_secret):.2f}]")
print(f"Extracted secret range (parameter-free): [{np.min(extracted_secret_alt):.2f}, {np.max(extracted_secret_alt):.2f}]")

# Check for overflow/underflow
if np.any(watermarked_img < 0) or np.any(watermarked_img > 255):
    print("⚠️  WARNING: Watermarked image values outside [0,255] range!")
else:
    print("✅ Watermarked image values within valid range [0,255]")

print(f"\n=== BLIND STEGANOGRAPHY RECOMMENDATIONS ===")
print("1. Statistical scaling method works best for blind extraction")
print("2. Store minimal metadata (e.g., scaling method used) if possible")
print("3. Parameter-free extraction provides fallback but lower quality")
print("4. Consider using error correction codes for robustness")
print("5. Test with different image types to ensure generalizability")