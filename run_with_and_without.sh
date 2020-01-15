echo "Running with AD"
python3 main.py --filename results/simplest_test --use_ad
echo "Running without AD"
python3 main.py --filename results/simplest_test --load_results
