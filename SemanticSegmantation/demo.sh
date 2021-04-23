
model_path=$1

ls demo_images/* | while read fn
do
    echo $fn
    python test_seg.py $fn $model_path
done
