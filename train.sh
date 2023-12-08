cd src

python main.py --multi_class --batch_size 8
# python main.py --multi_class --batch_size 8 --select_green
python main.py --multi_class --batch_size 8 --clahe
# python main.py --multi_class --batch_size 8 --clahe --select_green

python main.py --multi_class --batch_size 8 --apply_cbam
# python main.py --multi_class --batch_size 8 --select_green --apply_cbam
python main.py --multi_class --batch_size 8 --clahe --apply_cbam
# python main.py --multi_class --batch_size 8 --clahe --select_green --apply_cbam
