import struct
import gzip
import os

def estimate_actual_uncompressed_size(filename, sample_size=1024*1024):
    """
    Estimate the actual uncompressed size by sampling compression ratio.
    This works around the 4GB limitation of the gzip ISIZE field.
    """
    # Get the ISIZE field value (may be wrapped)
    with open(filename, 'rb') as f:
        f.seek(-4, 2)
        size_bytes = f.read(4)
        isize_value = struct.unpack('<I', size_bytes)[0]
    
    compressed_size = os.path.getsize(filename)
    
    print(f"Compressed file size: {compressed_size:,} bytes ({compressed_size / (1024**3):.2f} GB)")
    print(f"ISIZE field value: {isize_value:,} bytes ({isize_value / (1024**3):.2f} GB)")
    
    # Sample the beginning of the file to estimate compression ratio
    try:
        with gzip.open(filename, 'rb') as gz_file:
            # Read a sample from the beginning
            sample_compressed_pos = gz_file.fileobj.tell()
            sample_data = gz_file.read(sample_size)
            sample_uncompressed_size = len(sample_data)
            sample_compressed_size = gz_file.fileobj.tell() - sample_compressed_pos
            
            if sample_compressed_size > 0 and sample_uncompressed_size > 0:
                compression_ratio = sample_uncompressed_size / sample_compressed_size
                estimated_uncompressed = compressed_size * compression_ratio
                
                print(f"\nSample analysis (first {sample_size:,} bytes):")
                print(f"  Sample compressed: {sample_compressed_size:,} bytes")
                print(f"  Sample uncompressed: {sample_uncompressed_size:,} bytes")
                print(f"  Compression ratio: {compression_ratio:.2f}:1")
                print(f"  Estimated total uncompressed: {estimated_uncompressed:,} bytes ({estimated_uncompressed / (1024**3):.2f} GB)")
                
                # Check if ISIZE is likely wrapped
                max_32bit = 2**32
                if estimated_uncompressed > max_32bit:
                    wraps = int(estimated_uncompressed // max_32bit)
                    expected_isize = int(estimated_uncompressed % max_32bit)
                    print(f"\nISIZE Analysis:")
                    print(f"  Estimated wraps around 4GB: {wraps}")
                    print(f"  Expected ISIZE value: {expected_isize:,}")
                    print(f"  Actual ISIZE value: {isize_value:,}")
                    print(f"  Difference: {abs(expected_isize - isize_value):,}")
                
                return estimated_uncompressed
            else:
                print("Could not read sample data for estimation")
                return None
                
    except Exception as e:
        print(f"Error during estimation: {e}")
        return None

# Test with your file
if __name__ == "__main__":
    filename = "ol_dump_editions_latest.txt.gz"
    estimate_actual_uncompressed_size(filename) 