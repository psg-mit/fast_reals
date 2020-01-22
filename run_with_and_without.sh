echo "Running with AD"
file=results/junk
python3 main.py --filename $file --use_ad
echo "Running without AD"
python3 main.py --filename $file --load_results
