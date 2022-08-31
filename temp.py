pipeline = Pipeline(train_data=train_data,
                    eval_data=eval_data,
                    batch_size=4,
                    ops=[
                        LambdaOp(fn=lambda: 0, outputs="z"),
                        Repeat(AddOne(inputs="z", outputs="z"), repeat=5)
                    ])



pipeline = Pipeline(train_data=train_data,
                    eval_data=eval_data,
                    batch_size=4,
                    ops=[
                        LambdaOp(fn=lambda: 0, outputs="z"),
                        Repeat(AddOne(inputs="z", outputs="z"),
                               repeat=lambda z: z < 6.5)
                    ])


network = fe.Network(ops=[
    ModelOp(inputs="image", model=model, outputs="pred_segment"),
    Dice(inputs=("pred_segment", "mask"),
         outputs="dice_loss",
         sample_average=True,
         channel_average=True,
         negate=True),
    Repeat(AddOne(inputs="z", outputs="z"), repeat=5)
    UpdateOp(model=model, loss_name="dice_loss")
])


network = fe.Network(ops=[
    ModelOp(inputs="image", model=model, outputs="pred_segment"),
    Dice(inputs=("pred_segment", "mask"),
         outputs="dice_loss",
         sample_average=True,
         channel_average=True,
         negate=True),
    Repeat(AddOne(inputs="z", outputs="z"),
                               repeat=lambda z: z < 6.5),
    UpdateOp(model=model, loss_name="dice_loss")
])
