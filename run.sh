set -u
set -o pipefail

a=0
while [[ $a -lt 7190977 ]]
do
 from=$a
 to=$(( $a+2000000 ))

 echo $from, $to
 python -m deephack.extract --input "data/train.txt" --output "data/features/block_$from" --lower $from --upper $to &> "logs/block_$from" & 
 a=$to
done
