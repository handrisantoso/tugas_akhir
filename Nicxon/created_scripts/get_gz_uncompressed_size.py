import struct
import os

def get_uncompressed_size_gz(filename):
    """
    Get the uncompressed size from gzip file's ISIZE field.
    Note: Due to gzip format limitation, this is only accurate for files < 4GB.
    For larger files, the value wraps around (size % 2^32).
    """
    with open(filename, 'rb') as f:
        f.seek(-4, 2)  # Go to last 4 bytes
        size_bytes = f.read(4)
        size = struct.unpack('<I', size_bytes)[0]
        
        # Get compressed file size for comparison
        compressed_size = os.path.getsize(filename)
        
        print(f"Compressed file size: {compressed_size:,} bytes ({compressed_size / (1024**3):.2f} GB)")
        print(f"ISIZE field value: {size:,} bytes ({size / (1024**3):.2f} GB)")
        
        # Check if the uncompressed size might be wrapped around
        if compressed_size > 4 * 1024**3:  # If compressed > 4GB
            print("\n⚠️  WARNING: This gzip file is very large (>4GB compressed).")
            print("   The uncompressed size is likely much larger than reported.")
            print("   The gzip ISIZE field can only store values up to 4GB.")
            print("   Actual uncompressed size = ISIZE + (n × 4GB) where n is unknown.")
            
            # Estimate possible actual size
            estimated_min = size + 4 * 1024**3  # At least one wrap-around
            print(f"   Estimated minimum actual size: {estimated_min:,} bytes ({estimated_min / (1024**3):.2f} GB)")
        
        return size

print(get_uncompressed_size_gz("ol_dump_editions_latest.txt.gz"))