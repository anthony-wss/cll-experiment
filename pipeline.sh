TWCC_CLI_CMD=/tmp2/b09902033/pcll/bin/twccli
EXPERIMENT_CMD="cd cll-experiment; $1"
LOG_FILE=ccs_res_`date +%s`.log

echo "1. Creating CCS"      # 建立開發型容器
$TWCC_CLI_CMD mk ccs -itype "Pytorch" -img "pytorch-23.02-py3:latest" -gpu 1 -wait -json > $LOG_FILE

CCS_ID=$(cat $LOG_FILE | jq '.id')
echo "2. CCS ID:" $CCS_ID   # 開發型容器 ID

# echo "3. Checking GPU"      # 確認 GPU 狀態
# ssh -t -o "StrictHostKeyChecking=no" `$TWCC_CLI_CMD ls ccs -gssh -s $CCS_ID` "/bin/bash --login -c nvidia-smi"

echo "4. RUN GPU"           # 執行運算程式
ssh -t -o "StrictHostKeyChecking=no" `$TWCC_CLI_CMD ls ccs -gssh -s $CCS_ID` "/bin/bash --login -c '$EXPERIMENT_CMD'"
# 可依據您的程式，修改 "cd gpu-burn;/bin/bash --login -c './gpu_burn 150'"

echo "5. GC GPU"            # 刪除開發型容器
$TWCC_CLI_CMD rm ccs -f -s $CCS_ID

echo "6. Checking CCS"      # 檢視容器狀態
$TWCC_CLI_CMD ls ccs
rm $LOG_FILE
