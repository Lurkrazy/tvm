# yolo
python3 test_batch_auto_scheduler_dynamic_gradient_search_conv2d.py yolo 2>&1 | tee run_time1.log
mkdir -p run_time1_yolo
mv cuda* *log run_time1_yolo

python3 test_batch_auto_scheduler_dynamic_gradient_search_conv2d.py yolo 2>&1 | tee run_time2.log
mkdir -p run_time2_yolo
mv cuda* *log run_time2_yolo

python3 test_batch_auto_scheduler_dynamic_gradient_search_conv2d.py yolo 2>&1 | tee run_time3.log
mkdir -p run_time3_yolo
mv cuda* *log run_time3_yolo

# resnet
python3 test_batch_auto_scheduler_dynamic_gradient_search_conv2d.py resnet 2>&1 | tee run_time1.log
mkdir -p run_time1_resnet
mv cuda* *log run_time1_resnet

python3 test_batch_auto_scheduler_dynamic_gradient_search_conv2d.py resnet 2>&1 | tee run_time2.log
mkdir -p run_time2_resnet
mv cuda* *log run_time2_resnet

python3 test_batch_auto_scheduler_dynamic_gradient_search_conv2d.py resnet 2>&1 | tee run_time3.log
mkdir -p run_time3_resnet
mv cuda* *log run_time3_resnet


# matmul
python3 test_batch_auto_scheduler_dynamic_gradient_search_matmul.py 2>&1 | tee run_time1.log
mkdir -p run_time1
mv cuda* *log run_time1

python3 test_batch_auto_scheduler_dynamic_gradient_search_matmul.py 2>&1 | tee run_time2.log
mkdir -p run_time2
mv cuda* *log run_time2

python3 test_batch_auto_scheduler_dynamic_gradient_search_matmul.py 2>&1 | tee run_time3.log
mkdir -p run_time3
mv cuda* *log run_time3
