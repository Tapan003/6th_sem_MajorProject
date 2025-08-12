import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.cluster import KMeans
from scipy import stats

class BlindDWTSteganography:
    def __init__(self, wavelet='db8'):
        self.wavelet = wavelet
        
    def method_1_self_referencing(self, secret_img, cover_img):
        """
        Method 1: Self-Referencing Blind Extraction
        Uses relationships between different subbands for parameter estimation
        """
        secret_float = np.float32(secret_img)
        cover_float = np.float32(cover_img)
        
        # Get DWT coefficients
        secret_LL, (secret_LH, secret_HL, secret_HH) = pywt.dwt2(secret_float, self.wavelet)
        cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_float, self.wavelet)
        
        # Embedding with self-referencing markers
        # Use statistical properties of cover's LL to determine scaling
        cover_LL_stats = {
            'mean': np.mean(cover_LL),
            'std': np.std(cover_LL),
            'range': np.max(cover_LL) - np.min(cover_LL)
        }
        
        # Scale secret based on cover_LL properties (which we can estimate from watermarked)
        scaling_factor = cover_LL_stats['std'] * 0.1
        scaled_secret_LL = (secret_LL - np.mean(secret_LL)) / np.std(secret_LL) * scaling_factor + cover_LL_stats['mean']
        scaled_secret_LH = secret_LH * 0.05  # Fixed small scaling for LH
        
        # Embedding
        embedded_LL = cover_LL.copy()
        embedded_LH = cover_LH.copy()
        embedded_HL = scaled_secret_LL
        embedded_HH = cover_HH + scaled_secret_LH  # Additive embedding
        
        embedded_coeffs = (embedded_LL, (embedded_LH, embedded_HL, embedded_HH))
        watermarked_img = pywt.idwt2(embedded_coeffs, self.wavelet)
        
        return watermarked_img, cover_LL_stats
    
    def extract_method_1(self, watermarked_img):
        """Extract using self-referencing (estimates parameters from watermarked image)"""
        wm_coeffs = pywt.dwt2(watermarked_img, self.wavelet)
        wm_LL, (wm_LH, wm_HL, wm_HH) = wm_coeffs
        
        # Estimate original cover LL statistics from watermarked LL
        # (This works because we didn't modify LL during embedding)
        estimated_cover_stats = {
            'mean': np.mean(wm_LL),
            'std': np.std(wm_LL),
        }
        
        # Extract secret LL from HL subband
        scaling_factor = estimated_cover_stats['std'] * 0.1
        normalized_secret_LL = (wm_HL - estimated_cover_stats['mean']) / scaling_factor
        
        # Extract secret LH from HH (assuming we know the fixed scaling)
        # This is the limitation - we need to know or estimate this
        estimated_secret_LH = wm_HH / 0.05  # Reverse the 0.05 scaling
        
        # Reconstruct with zeros for missing subbands
        extracted_coeffs = (normalized_secret_LL, 
                          (estimated_secret_LH, 
                           np.zeros_like(wm_HL), 
                           np.zeros_like(wm_HH)))
        
        extracted_secret = pywt.idwt2(extracted_coeffs, self.wavelet)
        return extracted_secret
    
    def method_2_clustering_based(self, secret_img, cover_img):
        """
        Method 2: Clustering-Based Coefficient Separation
        Uses clustering to separate cover and secret coefficients
        """
        secret_float = np.float32(secret_img)
        cover_float = np.float32(cover_img)
        
        secret_LL, (secret_LH, secret_HL, secret_HH) = pywt.dwt2(secret_float, self.wavelet)
        cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_float, self.wavelet)
        
        # Create mixed coefficients that can be separated by clustering
        # Method: Quantize secret coefficients to specific ranges
        def quantize_for_clustering(coeffs, levels=4):
            """Quantize coefficients to discrete levels for clustering"""
            min_val, max_val = np.min(coeffs), np.max(coeffs)
            quantized = np.round((coeffs - min_val) / (max_val - min_val) * (levels-1)) * (max_val - min_val) / (levels-1) + min_val
            return quantized
        
        # Quantize secret for easier clustering
        quantized_secret_LL = quantize_for_clustering(secret_LL)
        
        # Embed by replacing certain coefficients entirely
        embedded_HL = cover_HL.copy()
        # Replace every 4th coefficient with quantized secret
        mask = np.zeros_like(cover_HL, dtype=bool)
        mask[::4, ::4] = True
        embedded_HL[mask] = quantized_secret_LL.flatten()[:np.sum(mask)]
        
        embedded_coeffs = (cover_LL, (cover_LH, embedded_HL, cover_HH))
        watermarked_img = pywt.idwt2(embedded_coeffs, self.wavelet)
        
        return watermarked_img, mask
    
    def extract_method_2(self, watermarked_img, mask):
        """Extract using clustering (needs embedding mask)"""
        wm_coeffs = pywt.dwt2(watermarked_img, self.wavelet)
        wm_LL, (wm_LH, wm_HL, wm_HH) = wm_coeffs
        
        # Extract coefficients using the mask
        secret_coeffs = wm_HL[mask]
        
        # Reconstruct secret LL subband
        secret_LL_shape = (wm_LL.shape[0], wm_LL.shape[1])
        extracted_secret_LL = np.zeros(secret_LL_shape)
        extracted_secret_LL.flat[:len(secret_coeffs)] = secret_coeffs
        
        extracted_coeffs = (extracted_secret_LL,
                          (np.zeros_like(wm_LH),
                           np.zeros_like(wm_HL), 
                           np.zeros_like(wm_HH)))
        
        extracted_secret = pywt.idwt2(extracted_coeffs, self.wavelet)
        return extracted_secret
    
    def method_3_statistical_modeling(self, secret_img, cover_img):
        """
        Method 3: Statistical Model-Based Embedding
        Uses known statistical distributions for blind extraction
        """
        secret_float = np.float32(secret_img)
        cover_float = np.float32(cover_img)
        
        secret_LL, (secret_LH, secret_HL, secret_HH) = pywt.dwt2(secret_float, self.wavelet)
        cover_LL, (cover_LH, cover_HL, cover_HH) = pywt.dwt2(cover_float, self.wavelet)
        
        # Model coefficients as following specific distributions
        # Natural images: LL follows Gaussian, others follow Laplacian
        
        def fit_to_laplacian(coeffs):
            """Scale coefficients to follow standard Laplacian distribution"""
            # Laplacian has specific mean=0, and scale parameter
            target_scale = 10.0  # Fixed scale for blind extraction
            current_scale = np.std(coeffs) * np.sqrt(2)  # Laplacian scale estimation
            if current_scale > 0:
                scaled = coeffs * (target_scale / current_scale)
            else:
                scaled = coeffs
            return scaled, target_scale
        
        # Scale secret to follow known distribution
        scaled_secret_LH, lh_scale = fit_to_laplacian(secret_LH)
        
        # Embedding in frequency domain with known distribution
        embedded_coeffs = (cover_LL, 
                         (cover_LH + scaled_secret_LH,  # Additive
                          cover_HL, 
                          cover_HH))
        
        watermarked_img = pywt.idwt2(embedded_coeffs, self.wavelet)
        return watermarked_img, lh_scale
    
    def extract_method_3(self, watermarked_img, original_cover_img=None):
        """Extract using statistical modeling (truly blind if no cover available)"""
        wm_coeffs = pywt.dwt2(watermarked_img, self.wavelet)
        wm_LL, (wm_LH, wm_HL, wm_HH) = wm_coeffs
        
        if original_cover_img is not None:
            # If cover available, subtract it
            cover_coeffs = pywt.dwt2(original_cover_img, self.wavelet)
            cover_LL, (cover_LH, cover_HL, cover_HH) = cover_coeffs
            extracted_secret_LH = wm_LH - cover_LH
        else:
            # Blind extraction: assume secret follows known distribution
            # Use high-pass filtering to isolate embedded signal
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(wm_LH, -1, kernel)
            
            # Statistical separation based on distribution assumptions
            threshold = np.std(filtered) * 0.5
            extracted_secret_LH = np.where(np.abs(filtered) > threshold, filtered, 0)
        
        # Reverse the known scaling
        target_scale = 10.0  # The fixed scale we used
        extracted_secret_LH = extracted_secret_LH / target_scale * np.sqrt(2)
        
        extracted_coeffs = (np.zeros_like(wm_LL),
                          (extracted_secret_LH,
                           np.zeros_like(wm_HL),
                           np.zeros_like(wm_HH)))
        
        extracted_secret = pywt.idwt2(extracted_coeffs, self.wavelet)
        return extracted_secret

# Demo usage
def demonstrate_methods():
    # Create sample images
    cover_img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    secret_img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    stego = BlindDWTSteganography()
    
    print("=== METHOD COMPARISON ===")
    
    # Method 1: Self-referencing
    print("\n1. Self-Referencing Method:")
    wm1, stats1 = stego.method_1_self_referencing(secret_img, cover_img)
    extracted1 = stego.extract_method_1(wm1)
    psnr1 = 20 * np.log10(255.0 / np.sqrt(np.mean((secret_img - np.clip(extracted1, 0, 255))**2)))
    print(f"   PSNR: {psnr1:.2f} dB")
    print(f"   Truly blind: YES (estimates parameters)")
    
    # Method 2: Clustering
    print("\n2. Clustering Method:")
    wm2, mask2 = stego.method_2_clustering_based(secret_img, cover_img)
    extracted2 = stego.extract_method_2(wm2, mask2)
    psnr2 = 20 * np.log10(255.0 / np.sqrt(np.mean((secret_img - np.clip(extracted2, 0, 255))**2)))
    print(f"   PSNR: {psnr2:.2f} dB")
    print(f"   Truly blind: NO (needs embedding mask)")
    
    # Method 3: Statistical modeling
    print("\n3. Statistical Modeling Method:")
    wm3, scale3 = stego.method_3_statistical_modeling(secret_img, cover_img)
    extracted3_blind = stego.extract_method_3(wm3)  # Truly blind
    extracted3_cover = stego.extract_method_3(wm3, cover_img)  # With cover
    
    psnr3_blind = 20 * np.log10(255.0 / np.sqrt(np.mean((secret_img - np.clip(extracted3_blind, 0, 255))**2)))
    psnr3_cover = 20 * np.log10(255.0 / np.sqrt(np.mean((secret_img - np.clip(extracted3_cover, 0, 255))**2)))
    
    print(f"   PSNR (blind): {psnr3_blind:.2f} dB")
    print(f"   PSNR (with cover): {psnr3_cover:.2f} dB")
    print(f"   Truly blind: YES (but lower quality)")

if __name__ == "__main__":
    demonstrate_methods()

# === TRADE-OFFS ANALYSIS ===
print("\n" + "="*50)
print("BLIND EXTRACTION TRADE-OFFS:")
print("="*50)

trade_offs = {
    "Method": ["Parameter-Based", "Self-Referencing", "Statistical Model", "Pure Assumption"],
    "Blind Level": ["Semi-blind", "Truly blind", "Truly blind", "Truly blind"],
    "Quality": ["High", "Medium", "Low-Medium", "Low"],
    "Robustness": ["High", "Medium", "Low", "Very Low"],
    "Storage Needed": ["Few parameters", "None", "None", "None"],
    "Complexity": ["Low", "Medium", "High", "Low"]
}

for i, method in enumerate(trade_offs["Method"]):
    print(f"\n{method}:")
    for key, values in trade_offs.items():
        if key != "Method":
            print(f"  {key}: {values[i]}")

print("\n" + "="*50)
print("RECOMMENDATIONS:")
print("="*50)
print("1. For highest quality: Use minimal parameters (mean, std only)")
print("2. For true blind: Self-referencing with subband relationships")
print("3. For robustness: Statistical modeling with error correction")
print("4. For stealth: Parameter-free with multiple extraction attempts")
print("5. Consider hybrid: Store 1-2 bytes of critical parameters")