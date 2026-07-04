import json
import os
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="Input file (misal: import_with_captions.json)")
    parser.add_argument("--out", dest="out", required=True, help="Output file untuk LLaMA-Factory (misal: train_vlm.json)")
    args = parser.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        ls_data = json.load(f)

    vlm_dataset = []

    # Variasi prompt pertanyaan untuk melatih model agar lebih dinamis
    prompts = [
        "<image>\nJelaskan kondisi jalan di depan beserta lokasinya.",
        "<image>\nApa bahaya yang terlihat pada gambar ini dan di mana letaknya?",
        "<image>\nBerikan peringatan ADAS dan koordinat objek untuk situasi ini.",
        "<image>\nAnalisis kerusakan jalan dari perspektif pengemudi beserta lokasinya."
    ]

    for task in ls_data:
        # 1. Pastikan path gambar absolut agar LLaMA-Factory tidak error
        image_path = task["data"]["image"] 

        predictions = task.get("predictions", [])
        if not predictions:
            continue
        
        result = predictions[0].get("result", [])
        
        combined_responses = []
        current_box_str = ""

        # 2. Pasangkan Bounding Box dengan Caption-nya
        for r in result:
            if r.get("type") == "rectanglelabels":
                val = r["value"]
                
                # Mengubah format persentase Label Studio (0-100) 
                # Menjadi format standar VLM (0-1000)
                xmin = int(val["x"] * 10)
                ymin = int(val["y"] * 10)
                xmax = int((val["x"] + val["width"]) * 10)
                ymax = int((val["y"] + val["height"]) * 10)
                
                # Format yang mudah dibaca oleh regex saat inference nanti
                current_box_str = f"[{xmin}, {ymin}, {xmax}, {ymax}]"
                
            elif r.get("type") == "textarea":
                text_list = r.get("value", {}).get("text", [])
                if text_list and current_box_str:
                    caption = text_list[0]
                    # Menyatukan teks peringatan ADAS dengan angka koordinat
                    combined_sentence = f"{caption} Koordinat: {current_box_str}"
                    combined_responses.append(combined_sentence)
                    
                    # Kosongkan untuk membaca objek berikutnya di gambar yang sama
                    current_box_str = ""

        if not combined_responses:
            continue

        # 3. Format akhir menjadi satu paragraf panjang jika ada banyak objek
        full_response = " ".join(combined_responses)

        entry = {
            "id": str(task["data"].get("id", random.randint(1000, 99999))),
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": random.choice(prompts) 
                },
                {
                    "from": "gpt",
                    "value": full_response
                }
            ]
        }
        vlm_dataset.append(entry)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(vlm_dataset, f, ensure_ascii=False, indent=2)

    print(f"Konversi VLM Selesai! {len(vlm_dataset)} gambar siap dilatih.")
    print(f"File dataset LLaMA-Factory disimpan di: {args.out}")

if __name__ == "__main__":
    main()