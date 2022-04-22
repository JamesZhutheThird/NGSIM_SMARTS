There are three main python programs. You can simply run each program to get the final envision in your browser.

`prepare_all_in_one.py` generates experts trajectory and observation features in `experts.npy`
`train_all_in_one.py` runs pre-train stage and finetune stage by default (setting `train_flag = True`). And checkpoints, results  and logs are saved in `./output/*` folder. You can load checkpoint and continue training by setting `continue_flag = True`, or you may just want to plot results by setting `plot_flag = True`.
`eval_all_in_one.py` uses the model trained before to run envision scripts, open `localhost:8081` in your browser to check the envision.

Copyright © 2019-2022 James Zhu Ⅲ All Rights Reserved
