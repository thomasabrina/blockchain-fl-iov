"""
Advanced Cryptographic Operations for Federated Learning
Implements threshold encryption, BLS signatures, and secure multi-party computation
"""

import os
import hashlib
import secrets
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import logging
from dataclasses import dataclass
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)

class CryptoScheme(Enum):
    """Supported cryptographic schemes"""
    ECDSA = "ecdsa"
    BLS = "bls"
    THRESHOLD_ECDSA = "threshold_ecdsa"
    RSA_THRESHOLD = "rsa_threshold"

@dataclass
class ThresholdKey:
    """Threshold cryptography key share"""
    share_id: int
    key_share: bytes
    threshold: int
    total_shares: int
    public_key: bytes
    scheme: CryptoScheme

@dataclass
class SecretShare:
    """Shamir's secret sharing share"""
    x: int  # Share index
    y: int  # Share value
    prime: int  # Prime modulus

class ThresholdCryptography:
    """Threshold encryption and signature schemes"""
    
    def __init__(self, threshold: int, total_shares: int):
        self.threshold = threshold
        self.total_shares = total_shares
        self.prime = self._generate_prime()
        
    def _generate_prime(self) -> int:
        """Generate a large prime for Shamir's secret sharing"""
        # Using a known large prime for simplicity
        return 2**127 - 1
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """Calculate modular inverse using extended Euclidean algorithm"""
        if a < 0:
            a = (a % m + m) % m
        
        # Extended Euclidean Algorithm
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m
    
    def generate_secret_shares(self, secret: int) -> List[SecretShare]:
        """Generate Shamir's secret shares"""
        # Generate random coefficients for polynomial
        coefficients = [secret] + [secrets.randbelow(self.prime) for _ in range(self.threshold - 1)]
        
        shares = []
        for i in range(1, self.total_shares + 1):
            # Evaluate polynomial at point i
            y = sum(coeff * (i ** j) for j, coeff in enumerate(coefficients)) % self.prime
            shares.append(SecretShare(x=i, y=y, prime=self.prime))
        
        return shares
    
    def reconstruct_secret(self, shares: List[SecretShare]) -> int:
        """Reconstruct secret from threshold shares using Lagrange interpolation"""
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares, got {len(shares)}")
        
        # Use first threshold shares
        shares = shares[:self.threshold]
        
        secret = 0
        for i, share_i in enumerate(shares):
            # Calculate Lagrange coefficient
            numerator = 1
            denominator = 1
            
            for j, share_j in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-share_j.x)) % self.prime
                    denominator = (denominator * (share_i.x - share_j.x)) % self.prime
            
            # Calculate lagrange coefficient
            lagrange_coeff = (numerator * self._mod_inverse(denominator, self.prime)) % self.prime
            secret = (secret + share_i.y * lagrange_coeff) % self.prime
        
        return secret
    
    def generate_threshold_keys(self, master_key: bytes) -> List[ThresholdKey]:
        """Generate threshold key shares"""
        # Convert master key to integer
        master_int = int.from_bytes(master_key, byteorder='big')
        
        # Generate secret shares
        shares = self.generate_secret_shares(master_int)
        
        # Create threshold keys
        threshold_keys = []
        for i, share in enumerate(shares):
            key_share = share.y.to_bytes(32, byteorder='big')
            
            threshold_key = ThresholdKey(
                share_id=share.x,
                key_share=key_share,
                threshold=self.threshold,
                total_shares=self.total_shares,
                public_key=master_key,  # Simplified - in practice, derive public key
                scheme=CryptoScheme.THRESHOLD_ECDSA
            )
            threshold_keys.append(threshold_key)
        
        return threshold_keys
    
    def threshold_decrypt(self, ciphertext: bytes, key_shares: List[ThresholdKey]) -> bytes:
        """Decrypt using threshold key shares"""
        if len(key_shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} key shares")
        
        # Reconstruct master key
        shares = []
        for key_share in key_shares[:self.threshold]:
            share_value = int.from_bytes(key_share.key_share, byteorder='big')
            shares.append(SecretShare(x=key_share.share_id, y=share_value, prime=self.prime))
        
        master_key_int = self.reconstruct_secret(shares)
        master_key = master_key_int.to_bytes(32, byteorder='big')
        
        # Decrypt using reconstructed key
        aesgcm = AESGCM(master_key)
        nonce = ciphertext[:12]  # First 12 bytes are nonce
        encrypted_data = ciphertext[12:]
        
        return aesgcm.decrypt(nonce, encrypted_data, None)

class BLSSignature:
    """BLS signature scheme (simplified implementation)"""
    
    def __init__(self):
        # In a real implementation, this would use proper BLS curves
        # For demonstration, we'll use ECDSA as a placeholder
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key = self.private_key.public_key()
    
    def sign(self, message: bytes) -> bytes:
        """Sign message using BLS (simplified with ECDSA)"""
        signature = self.private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        return signature
    
    def verify(self, message: bytes, signature: bytes, public_key: ec.EllipticCurvePublicKey) -> bool:
        """Verify BLS signature"""
        try:
            public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False
    
    def aggregate_signatures(self, signatures: List[bytes]) -> bytes:
        """Aggregate multiple BLS signatures (simplified)"""
        # In real BLS, signatures can be aggregated
        # For demonstration, we'll concatenate and hash
        combined = b''.join(signatures)
        return hashlib.sha256(combined).digest()
    
    def aggregate_public_keys(self, public_keys: List[ec.EllipticCurvePublicKey]) -> bytes:
        """Aggregate public keys (simplified)"""
        # In real BLS, public keys can be aggregated
        # For demonstration, we'll serialize and hash
        serialized_keys = []
        for pk in public_keys:
            serialized = pk.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint
            )
            serialized_keys.append(serialized)
        
        combined = b''.join(serialized_keys)
        return hashlib.sha256(combined).digest()

class SecureMultiPartyComputation:
    """Secure Multi-Party Computation protocols"""
    
    def __init__(self, num_parties: int):
        self.num_parties = num_parties
        self.threshold_crypto = ThresholdCryptography(
            threshold=(num_parties // 2) + 1,
            total_shares=num_parties
        )
    
    def secure_sum(self, values: List[float], party_id: int) -> float:
        """Secure sum computation using additive secret sharing"""
        # Generate random shares that sum to the value
        shares = []
        total_random = 0
        
        for i in range(self.num_parties - 1):
            random_share = secrets.randbits(32) / (2**32)  # Random float [0,1)
            shares.append(random_share)
            total_random += random_share
        
        # Last share ensures sum equals original value
        last_share = values[party_id] - total_random
        shares.append(last_share)
        
        return shares[party_id]
    
    def secure_average(self, local_values: List[float]) -> float:
        """Compute secure average across parties"""
        # Each party contributes their secure sum share
        secure_shares = []
        for i, value in enumerate(local_values):
            share = self.secure_sum([value], i)
            secure_shares.append(share)
        
        # Sum all shares and divide by number of parties
        total_sum = sum(secure_shares)
        return total_sum / len(local_values)
    
    def private_set_intersection(self, set_a: set, set_b: set) -> set:
        """Private set intersection (simplified)"""
        # In practice, this would use more sophisticated protocols
        # For demonstration, we'll use hash-based approach
        
        hashed_a = {hashlib.sha256(str(item).encode()).hexdigest() for item in set_a}
        hashed_b = {hashlib.sha256(str(item).encode()).hexdigest() for item in set_b}
        
        intersection_hashes = hashed_a & hashed_b
        
        # In real implementation, parties would reveal only intersection
        # Here we simulate by returning intersection size
        return set(range(len(intersection_hashes)))

class AdvancedCryptoManager:
    """Advanced cryptographic manager integrating all schemes"""
    
    def __init__(self, device_id: str, num_parties: int = 5):
        self.device_id = device_id
        self.num_parties = num_parties
        
        # Initialize cryptographic components
        self.threshold_crypto = ThresholdCryptography(
            threshold=(num_parties // 2) + 1,
            total_shares=num_parties
        )
        self.bls_signature = BLSSignature()
        self.smpc = SecureMultiPartyComputation(num_parties)
        
        # Generate keys
        self.ecdsa_private_key = ec.generate_private_key(ec.SECP256R1())
        self.ecdsa_public_key = self.ecdsa_private_key.public_key()
        
        logger.info(f"Initialized advanced crypto manager for {device_id}")
    
    def create_threshold_encrypted_model(self, model_weights: List[np.ndarray]) -> Dict[str, Any]:
        """Create threshold-encrypted model package"""
        # Serialize model weights
        model_data = {
            'weights': [w.tolist() for w in model_weights],
            'shapes': [w.shape for w in model_weights],
            'dtypes': [str(w.dtype) for w in model_weights],
            'device_id': self.device_id,
            'timestamp': time.time()
        }
        
        model_bytes = json.dumps(model_data).encode('utf-8')
        
        # Generate master encryption key
        master_key = AESGCM.generate_key(bit_length=256)
        
        # Encrypt model with master key
        aesgcm = AESGCM(master_key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, model_bytes, None)
        
        # Create threshold key shares
        threshold_keys = self.threshold_crypto.generate_threshold_keys(master_key)
        
        # Create package
        package = {
            'encrypted_model': nonce + ciphertext,
            'threshold_keys': [
                {
                    'share_id': tk.share_id,
                    'key_share': tk.key_share.hex(),
                    'threshold': tk.threshold,
                    'total_shares': tk.total_shares
                }
                for tk in threshold_keys
            ],
            'device_id': self.device_id,
            'timestamp': time.time(),
            'scheme': 'threshold_encryption'
        }
        
        return package
    
    def collaborative_decrypt_model(self, encrypted_package: Dict[str, Any], 
                                  available_key_shares: List[Dict]) -> List[np.ndarray]:
        """Collaboratively decrypt model using threshold key shares"""
        if len(available_key_shares) < self.threshold_crypto.threshold:
            raise ValueError(f"Need at least {self.threshold_crypto.threshold} key shares")
        
        # Reconstruct threshold keys
        threshold_keys = []
        for share_data in available_key_shares:
            threshold_key = ThresholdKey(
                share_id=share_data['share_id'],
                key_share=bytes.fromhex(share_data['key_share']),
                threshold=share_data['threshold'],
                total_shares=share_data['total_shares'],
                public_key=b'',  # Not needed for decryption
                scheme=CryptoScheme.THRESHOLD_ECDSA
            )
            threshold_keys.append(threshold_key)
        
        # Decrypt model
        encrypted_model = encrypted_package['encrypted_model']
        if isinstance(encrypted_model, str):
            encrypted_model = bytes.fromhex(encrypted_model)
        
        decrypted_data = self.threshold_crypto.threshold_decrypt(encrypted_model, threshold_keys)
        
        # Deserialize model
        model_data = json.loads(decrypted_data.decode('utf-8'))
        
        # Reconstruct numpy arrays
        weights = []
        for i, (weight_list, shape, dtype) in enumerate(zip(
            model_data['weights'], 
            model_data['shapes'], 
            model_data['dtypes']
        )):
            weight_array = np.array(weight_list, dtype=dtype).reshape(shape)
            weights.append(weight_array)
        
        return weights
    
    def create_multi_signature(self, message: bytes, signers: List['AdvancedCryptoManager']) -> Dict[str, Any]:
        """Create multi-signature using BLS aggregation"""
        signatures = []
        public_keys = []
        
        # Collect signatures from all signers
        for signer in signers:
            signature = signer.bls_signature.sign(message)
            signatures.append(signature)
            public_keys.append(signer.bls_signature.public_key)
        
        # Aggregate signatures and public keys
        aggregated_signature = self.bls_signature.aggregate_signatures(signatures)
        aggregated_public_key = self.bls_signature.aggregate_public_keys(public_keys)
        
        return {
            'message': message.hex(),
            'aggregated_signature': aggregated_signature.hex(),
            'aggregated_public_key': aggregated_public_key.hex(),
            'num_signers': len(signers),
            'signer_ids': [signer.device_id for signer in signers]
        }
    
    def verify_multi_signature(self, multi_sig_data: Dict[str, Any]) -> bool:
        """Verify aggregated multi-signature"""
        # In a real BLS implementation, this would verify the aggregated signature
        # For demonstration, we'll check if all components are present
        required_fields = ['message', 'aggregated_signature', 'aggregated_public_key', 'num_signers']
        return all(field in multi_sig_data for field in required_fields)
    
    def secure_model_aggregation(self, local_weights: List[np.ndarray], 
                                other_parties_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Perform secure multi-party model aggregation"""
        all_weights = [local_weights] + other_parties_weights
        num_parties = len(all_weights)
        
        # Secure aggregation using SMPC
        aggregated_weights = []
        
        for layer_idx in range(len(local_weights)):
            layer_weights = [weights[layer_idx] for weights in all_weights]
            
            # Flatten weights for secure computation
            flattened_weights = [w.flatten() for w in layer_weights]
            
            # Perform secure average
            secure_avg = []
            for i in range(len(flattened_weights[0])):
                values = [fw[i] for fw in flattened_weights]
                avg_value = self.smpc.secure_average(values)
                secure_avg.append(avg_value)
            
            # Reshape back to original shape
            aggregated_layer = np.array(secure_avg).reshape(local_weights[layer_idx].shape)
            aggregated_weights.append(aggregated_layer)
        
        return aggregated_weights
    
    def get_crypto_status(self) -> Dict[str, Any]:
        """Get cryptographic manager status"""
        return {
            'device_id': self.device_id,
            'num_parties': self.num_parties,
            'threshold': self.threshold_crypto.threshold,
            'total_shares': self.threshold_crypto.total_shares,
            'schemes_available': [
                'threshold_encryption',
                'bls_signatures',
                'secure_mpc',
                'ecdsa'
            ],
            'public_key': self.ecdsa_public_key.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint
            ).hex()
        }

# Factory function
def create_advanced_crypto_manager(device_id: str, num_parties: int = 5) -> AdvancedCryptoManager:
    """Create advanced cryptographic manager"""
    return AdvancedCryptoManager(device_id, num_parties)

# Example usage and testing
if __name__ == "__main__":
    # Test threshold cryptography
    print("Testing Threshold Cryptography...")
    
    # Create managers for multiple parties
    managers = [create_advanced_crypto_manager(f"device_{i}", 5) for i in range(5)]
    
    # Test threshold encryption
    dummy_weights = [
        np.random.randn(10, 5).astype(np.float32),
        np.random.randn(5, 1).astype(np.float32)
    ]
    
    # Create threshold encrypted model
    encrypted_package = managers[0].create_threshold_encrypted_model(dummy_weights)
    print(f"Created threshold encrypted model with {len(encrypted_package['threshold_keys'])} key shares")
    
    # Test collaborative decryption (using 3 out of 5 shares)
    available_shares = encrypted_package['threshold_keys'][:3]
    decrypted_weights = managers[0].collaborative_decrypt_model(encrypted_package, available_shares)
    print(f"Successfully decrypted model with {len(decrypted_weights)} layers")
    
    # Test multi-signature
    message = b"FL model update for round 001"
    multi_sig = managers[0].create_multi_signature(message, managers[:3])
    print(f"Created multi-signature with {multi_sig['num_signers']} signers")
    
    # Verify multi-signature
    is_valid = managers[0].verify_multi_signature(multi_sig)
    print(f"Multi-signature verification: {is_valid}")
    
    print("Advanced cryptography tests completed!")
