if [ ! -d "./logs" ]; then
  mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting_FEDformer" ]; then
  mkdir ./logs/LongForecasting_FEDformer
fi

seq_len=96
label_len=48
station_type=adaptive
features=M
gpu=0


for model_name in FEDformer; do
  for pred_len in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=$gpu python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_HDN \
      --root_path ./datasets/exchange_rate \
      --data_path exchange_rate.csv \
      --model_id exchange_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 8 \
      --dec_in 8 \
      --c_out 8 \
      --des 'Exp' \
      --station_lr 0.0005 \
      --learning_rate 0.00001 \
      --station_type $station_type \
      --max_period 6 \
      --itr 1 >logs/LongForecasting_FEDformer/$model_name'_exchange_rate_'$pred_len.log

    CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_HDN \
      --root_path ./datasets/electricity \
      --data_path electricity.csv \
      --model_id electricity_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --station_type $station_type \
      --learning_rate 0.0005 \
      --station_lr 0.0005 \
      --itr 1 >logs/LongForecasting_FEDformer/$model_name'_electricity_'$pred_len.log

    CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_HDN \
      --root_path ./datasets/traffic \
      --data_path traffic.csv \
      --model_id traffic_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --station_type $station_type \
      --station_lr 0.00001 \
      --learning_rate 0.001 \
      --itr 1 >logs/LongForecasting_FEDformer/$model_name'_traffic_'$pred_len.log
    
    CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_HDN \
      --root_path ./datasets/ETT-small \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --station_type $station_type \
      --learning_rate 0.0005 \
      --station_lr 0.01 \
      --top_k 1 \
      --itr 1 >logs/LongForecasting_FEDformer/$model_name'_Etth1_'$pred_len.log

    CUDA_VISIBLE_DEVICES=$gpu \
    python3 -u run_longExp.py \
      --is_training 1 \
      --exp Exp_HDN \
      --root_path ./datasets/ETT-small \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --station_type $station_type \
      --learning_rate 0.00001 \
      --station_lr 0.00005 \
      --itr 1 >logs/LongForecasting_FEDformer/$model_name'_Etth2_'$pred_len.log

    CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_HDN \
      --root_path ./datasets/ETT-small \
      --data_path ETTm1.csv \
      --model_id ETTm1_96_$pred_len \
      --model $model_name \
      --data ETTm1 \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --station_type $station_type \
      --learning_rate 0.0001 \
      --station_lr 0.00005 \
      --max_period 6 \
      --top_k 1 \
      --itr 1 >logs/LongForecasting_FEDformer/$model_name'_Ettm1_'$pred_len.log

    CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_HDN \
      --root_path ./datasets/ETT-small \
      --data_path ETTm2.csv \
      --model_id ETTm2_96_$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --station_type $station_type \
      --max_period 6 \
      --station_lr 0.00001 \
      --learning_rate 0.0001 \
      --top_k 1 \
      --itr 1 > logs/LongForecasting_FEDformer/$model_name'_Ettm2_'$pred_len.log


    CUDA_VISIBLE_DEVICES=$gpu \
    python -u run_longExp.py \
      --is_training 1 \
      --exp Exp_HDN \
      --root_path ./datasets/weather \
      --data_path weather.csv \
      --model_id weather_96_$pred_len \
      --model $model_name \
      --data custom \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --learning_rate 0.00005 \
      --station_lr 0.0005 \
      --station_type $station_type \
      --max_period 12 \
      --itr 1 >logs/LongForecasting_FEDformer/$model_name'_weather_'$pred_len.log

  done
done
