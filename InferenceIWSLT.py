import sys
import torch
import os

modelname = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])

checkpointfolder = 'results/IWSLT/checkpoints/'
ensemblemodelpath = checkpointfolder + 'ensemblemodel.pt'

for n in [10]:
    bestbleu = 0
    modelfolder = checkpointfolder + modelname + '/'
    bleufolder = 'results/IWSLT/BLEU/' + modelname + '/ensemble{}/'.format(n)
    os.system('mkdir -p {}'.format(bleufolder))

    bestepoch = start
    jlist = [j for j in range(start, end + 1)]

    for j in jlist:
        cpname = modelfolder + 'checkpoint{}.pt'.format(j)
        model = torch.load(cpname)
        for i in range(1, n):
            cpname2 = modelfolder + 'checkpoint{}.pt'.format(j - i)
            model2 = torch.load(cpname2)
            for param in model['model']:
                model['model'][param].add_(model2['model'][param])
            del model2
        for param in model['model']:
            model['model'][param].div_(float(n))
        torch.save(model, ensemblemodelpath)
        del model
        bleu = bleufolder + 'checkpoint{}_ensemble{}.out'.format(j, n)
        print('evaluating {}'.format(bleu))

        command = 'python generate.py data-bin/iwslt14.tokenized.de-en/ --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 --quiet | tee {}'.format(ensemblemodelpath,bleu)
        os.system(command)
        with open(bleu, 'r') as f:
            lines = f.read().splitlines()
            lastline = lines[-1].replace(',', '').split()
            if bestbleu < float(lastline[13]):
                bestbleu = float(lastline[13])
                bestepoch = j
            print('best bleu {} at epoch {}'.format(bestbleu, bestepoch))

    bestensemble = modelfolder + 'bestmodel_ensemble{}_epoch{}_{}.pt'.format(n, bestepoch-n+1, bestepoch)
    cpname = modelfolder + 'checkpoint{}.pt'.format(bestepoch)
    model = torch.load(cpname)
    for i in range(1, n):
        cpname2 = modelfolder + 'checkpoint{}.pt'.format(bestepoch - i)
        model2 = torch.load(cpname2)
        for param in model['model']:
            model['model'][param].add_(model2['model'][param])
        del model2
    for param in model['model']:
        model['model'][param].div_(float(n))
    torch.save(model, bestensemble)
    del model
    bleu = bleufolder + 'bestmodel_ensemble{}_epoch{}_{}.out'.format(n, bestepoch-n+1, bestepoch)
    print('evaluating {}'.format(bleu))
    command = 'python generate.py data-bin/iwslt14.tokenized.de-en/ --path {} --beam 4 --batch-size 128 --remove-bpe --lenpen 0.3 | tee {}'.format(bestensemble, bleu)
    os.system(command)
    command = './compound_split_bleu.sh {}'.format(bleu)
    os.system(command)
