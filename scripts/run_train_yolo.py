import sys
from hq_det.tools.train_yolo import MyTrainer
from hq_det.trainer import HQTrainerArguments


if __name__ == '__main__':
    trainer = MyTrainer(
        HQTrainerArguments(
            data_path=sys.argv[1],
            num_epoches=50,
            warmup_epochs=0,
            num_data_workers=8,
            lr0=1e-4,
            lr_min=1e-6,
            batch_size=4,
            device='cuda:0',
            checkpoint_interval=-1,
            model_argument={
                "model_path": sys.argv[2]
            },
            image_size=1024,
        )
    )
    trainer.run()