# today=`date`
# echo $today
# python3 -c 'import NB_WMain; NB_WMain.a1_encrypt()' >> 100a1_encrypt.txt
# python3 -c 'import NB_WMain; NB_WMain.a1_check(0)' >> last_a1_check.txt
# rm -r ./a1_ctxt ./a1_train

# today=`date`
# echo $today
# python3 -c 'import NB_WMain; NB_WMain.a2_encrypt()' >> 100a2_encrypt.txt
# python3 -c 'import NB_WMain; NB_WMain.a2_check(0)'>> last_a2_check.txt
# rm -r ./a2_ctxt ./a2_train

# today=`date`
# echo $today
# python3 -c 'import NB_WMain; NB_WMain.cancer_encrypt()' >> 200cancer_encrypt.txt
# python3 -c 'import NB_WMain; NB_WMain.cancer_check(0)'  >> last_cancer_check.txt
# rm -r ./cancer_ctxt ./cancer_train

today=`date`
echo $today
python3 -c 'import NB_WMain; NB_WMain.car_encrypt()' >> car_encrypt.txt
python3 -c 'import NB_WMain; NB_WMain.car_check(1)' >> last_car_check.txt
rm -r ./car_ctxt ./car_train
today=`date`
echo $today

today=`date`
echo $today
python3 -c 'import Exp_Main; Exp_Main.a1_encrypt()' >> 0913a1_encrypt.txt
python3 -c 'import Exp_Main; Exp_Main.a1_check(0)' >> 0913last_a1_check.txt
rm -r ./a1_ctxt ./a1_train

today=`date`
echo $today
python3 -c 'import Exp_Main; Exp_Main.a2_encrypt()' >> 0913a2_encrypt.txt
python3 -c 'import Exp_Main; Exp_Main.a2_check(0)'>> 0913last_a2_check.txt
rm -r ./a2_ctxt ./a2_train

today=`date`
echo $today
python3 -c 'import Exp_Main; Exp_Main.cancer_encrypt()' >> 0913cancer_encrypt.txt
python3 -c 'import Exp_Main; Exp_Main.cancer_check(0)'  >> 0913last_cancer_check.txt
rm -r ./cancer_ctxt ./cancer_train

today=`date`
echo $today
python3 -c 'import Exp_Main; Exp_Main.car_encrypt()' >> 0913car_encrypt.txt
python3 -c 'import Exp_Main; Exp_Main.car_check(0)' >> 0913last_car_check.txt
rm -r ./car_ctxt ./car_train
today=`date`
echo $today