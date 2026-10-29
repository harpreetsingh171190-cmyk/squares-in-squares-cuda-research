"""CLI: python -m packing_ai.cli analyze --csv inputs/measured.csv --out reports/run1"""
import argparse, json
from .experiments import load_experiments
from .advanced import data_quality,experiment_manifest,export_bundle

def main():
    parser=argparse.ArgumentParser(description='CUDA packing experiment analytics (no solver execution)')
    sub=parser.add_subparsers(dest='command',required=True)
    cmd=sub.add_parser('analyze');cmd.add_argument('--csv',required=True);cmd.add_argument('--out',required=True)
    args=parser.parse_args()
    df=load_experiments(args.csv)
    report=experiment_manifest(df,{'quality':data_quality(df)})
    export_bundle(df,report,args.out)
    print(json.dumps({'output':args.out,'quality':report['configuration']['quality'],'dataset_sha256':report['dataset_sha256']},indent=2))
if __name__=='__main__': main()
