from model import *
def fetch_classifier(args):
    if 'lstm' in args.model:
        model = ClassifierLSTM(args, input=args.input, output=args.output)
    elif 'gru' in args.model:
        model = ClassifierGRU(args, input=args.input, output=args.output)
    elif 'dcnn' in args.model:
        # print("dcnn")
        print(args)
        model = BenchmarkDCNN(args, input=args.input, output=args.activity_label_size)
    elif 'cnn2' in args.model:
        model = ClassifierCNN2D(args, output=args.output)
    elif 'cnn1' in args.model:
        model = ClassifierCNN1D(args, output=args.output)
    elif 'deepsense' in args.model:
        model = BenchmarkDeepSense(args, input=args.input, output=args.output)
    elif 'attn' in args.model:
        model = ClassifierAttn(args, input=args.input, output=args.output)
    else:
        model = None
    return model