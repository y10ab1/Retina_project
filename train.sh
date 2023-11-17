cd src

python main.py
python main.py --select_green
python main.py --clahe
python main.py --clahe --select_green

python main.py --apply_cbam
python main.py --select_green --apply_cbam
python main.py --clahe --apply_cbam
python main.py --clahe --select_green --apply_cbam
