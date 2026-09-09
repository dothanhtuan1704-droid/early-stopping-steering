import glob, json, os

ipynb_files = glob.glob('*.ipynb') + glob.glob('*/*.ipynb')
fixed_count = 0

print(f"Scanning {len(ipynb_files)} notebook files in repository...")

for fn in ipynb_files:
    try:
        with open(fn, 'r', encoding='utf-8') as f:
            content = f.read()

        modified = False

        # Fix pos_ref / neg_ref KeyError
        if "'pos_ref'" in content or '"pos_ref"' in content:
            # Replace direct index lookups with safe fallbacks
            content = content.replace("rec['pos_ref']", "rec.get('pos_ref', rec.get('right_answer', ''))")
            content = content.replace("rec['neg_ref']", "rec.get('neg_ref', rec.get('hallucinated_answer', ''))")
            content = content.replace("item['pos_ref']", "item.get('pos_ref', item.get('right_answer', ''))")
            content = content.replace("item['neg_ref']", "item.get('neg_ref', item.get('hallucinated_answer', ''))")
            modified = True

        # Fix Multi-GPU device mismatch if hook present
        if "out_tensor[:, -1, :] = out_tensor[:, -1, :] + (self.alpha * self.vector" in content:
            content = content.replace(
                "out_tensor[:, -1, :] = out_tensor[:, -1, :] + (self.alpha * self.vector.to(out_tensor.dtype))",
                "steer_vec = self.vector.to(device=out_tensor.device, dtype=out_tensor.dtype)\n            out_tensor[:, -1, :] = out_tensor[:, -1, :] + (self.alpha * steer_vec)"
            )
            content = content.replace(
                "out_tensor[:, -1, :] = out_tensor[:, -1, :] + (self.alpha * self.vector)",
                "steer_vec = self.vector.to(device=out_tensor.device, dtype=out_tensor.dtype)\n            out_tensor[:, -1, :] = out_tensor[:, -1, :] + (self.alpha * steer_vec)"
            )
            modified = True

        if modified:
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(content)
            fixed_count += 1
            print(f"  - Fixed notebook: {fn}")

    except Exception as e:
        print(f"  - Error checking {fn}: {e}")

print(f"\n[SUCCESS] Audit complete! Updated {fixed_count}/{len(ipynb_files)} notebooks.")
