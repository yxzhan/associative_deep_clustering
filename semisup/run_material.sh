
LOGFILE="./out.log"
rm $LOGFILE
nohup python3 material_train_eval2.py 1> $LOGFILE  2>/dev/null &