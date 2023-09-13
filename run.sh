today=`date`
echo $today
python3 -c 'import NB_WMain; NB_WMain.a1_encrypt()' >> a1_encrypt.txt
python3 -c 'import NB_WMain; NB_WMain.a1_check(0)' >> a1_check.txt
rm -r ./a1_ctxt ./a1_train

today=`date`
echo $today
python3 -c 'import NB_WMain; NB_WMain.a2_encrypt()' >> a2_encrypt.txt
python3 -c 'import NB_WMain; NB_WMain.a2_check(0)'>> a2_check.txt
rm -r ./a2_ctxt ./a2_train

today=`date`
echo $today
python3 -c 'import NB_WMain; NB_WMain.cancer_encrypt()' >> cancer_encrypt.txt
python3 -c 'import NB_WMain; NB_WMain.cancer_check(0)'  >> cancer_check.txt
rm -r ./cancer_ctxt ./cancer_train

today=`date`
echo $today
python3 -c 'import NB_WMain; NB_WMain.car_encrypt()' >> car_encrypt.txt
python3 -c 'import NB_WMain; NB_WMain.car_check(0)' >> car_check.txt
rm -r ./car_ctxt ./car_train