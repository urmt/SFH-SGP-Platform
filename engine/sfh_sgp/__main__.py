import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog='sfh-sgp', description='SFH-SGP Platform CLI')
    sub = parser.add_subparsers(dest='command')

    serve = sub.add_parser('serve', help='Start the API server')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8000)
    serve.add_argument('--reload', action='store_true', help='Auto-reload on code changes')

    embed = sub.add_parser('embed', help='Embed a signal from CSV')
    embed.add_argument('file', help='CSV file with a "value" column')
    embed.add_argument('--name', default='', help='Signal name/id')
    embed.add_argument('--transform', default='base',
                       choices=['base', 'reverse', 'swap', 'replay', 'stitch'])

    analyse = sub.add_parser('analyse', help='Run geometry analysis on signals in CSV')
    analyse.add_argument('file', help='CSV file with a "value" column (one signal per column, or column "signal_id","value")')

    run = sub.add_parser('experiment', help='Run all synthetic experiments')
    run.add_argument('--save', help='Save results to JSON file')

    args = parser.parse_args()

    if args.command == 'serve':
        _serve(args)
    elif args.command == 'embed':
        _embed(args)
    elif args.command == 'analyse':
        _analyse(args)
    elif args.command == 'experiment':
        _experiment(args)
    else:
        parser.print_help()
        sys.exit(1)


def _serve(args):
    import uvicorn
    print(f'SFH-SGP API server starting on http://{args.host}:{args.port}')
    print(f'  Docs:   http://{args.host}:{args.port}/docs')
    print(f'  Health: http://{args.host}:{args.port}/health')
    uvicorn.run('sfh_sgp.api.main:app', host=args.host, port=args.port,
                reload=args.reload)


def _embed(args):
    import numpy as np
    data = np.genfromtxt(args.file, delimiter=',', names=True)
    if 'value' in data.dtype.names:
        values = data['value']
    else:
        col = data.dtype.names[0]
        values = data[col]
    from .core.embedding import EmbeddingEngine
    from .core.transforms import TransformEngine
    engine = EmbeddingEngine()
    transformer = TransformEngine()
    x = np.asarray(values, dtype=float)
    if args.transform != 'base':
        x = transformer.apply(x, args.transform)
    result = engine.embed(x, signal_id=args.name, transform=args.transform)
    print(f'Signal:         {args.name or args.file}')
    print(f'Transform:      {args.transform}')
    print(f'Architecture:   {result.architecture}')
    print(f'm1 (ordinal_flow):             {result.m1:.6f}')
    print(f'm2 (half_correlation):         {result.m2:.6f}')
    print(f'm3 (signed_compressibility):   {result.m3:.6f}')
    print(f'm4 (amp_transition_asym):      {result.m4:.6f}')
    print(f'Embedding vector: [{result.m1:.4f}, {result.m2:.4f}, {result.m3:.4f}, {result.m4:.4f}]')


def _analyse(args):
    import numpy as np
    data = np.genfromtxt(args.file, delimiter=',', names=True)
    signals = []
    names = data.dtype.names
    if 'value' in names:
        col_names = ['value']
    else:
        col_names = names
    from .core.embedding import EmbeddingEngine
    from .core.geometry import GeometryAnalyser
    engine = EmbeddingEngine()
    geo = GeometryAnalyser()
    embeddings = []
    for col in col_names:
        x = np.asarray(data[col], dtype=float)
        if np.any(np.isnan(x)):
            continue
        result = engine.embed(x, signal_id=col)
        embeddings.append(result.vector)
    if len(embeddings) < 2:
        print('Need at least 2 valid signals for geometry analysis')
        sys.exit(1)
    result = geo.analyse(np.array(embeddings))
    print(f'Geometry Analysis ({len(embeddings)} signals)')
    print(f'  PC1 variance:           {result["pc1"]}')
    print(f'  Intrinsic dim (95%):    {result["dim95"]}')
    print(f'  Curvature:              {result["curvature"]}')
    print(f'  Neighbour purity:       {result["neighbor_purity"]}')
    print(f'  Geo-Euclidean corr:     {result["geo_euclidean_corr"]}')


def _experiment(args):
    from .core.embedding import EmbeddingEngine
    from .core.transforms import TransformEngine
    from .core.geometry import GeometryAnalyser
    from .core.experiments import ExperimentRunner
    from .io.synthetic import random_walk, regime_switch
    import numpy as np
    engine = EmbeddingEngine()
    t = TransformEngine()
    g = GeometryAnalyser()
    runner = ExperimentRunner(engine, t, g)
    n_signals = 50
    rw = [random_walk(seed=i) for i in range(n_signals)]
    rs = [regime_switch(seed=i) for i in range(n_signals)]
    all_sigs = rw + rs
    labels = [0] * n_signals + [1] * n_signals
    rw_embs = np.array([r.vector for r in engine.embed_many(rw)])
    rs_embs = np.array([r.vector for r in engine.embed_many(rs)])
    replay_sig = random_walk(seed=42)
    signal_set = {
        'classification_signals': all_sigs,
        'classification_labels': labels,
        'replay_signal': replay_sig,
    }
    results = runner.run_all(signal_set)
    results.append(runner.run_f004_rw_trend_overlap(rw_embs, rs_embs))
    for r in results:
        status = 'PASS' if r['passed'] else 'FAIL'
        print(f'  {r["experiment"]:45s}  [{status}]  {r["metrics"]}')
    if args.save:
        import json
        with open(args.save, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nResults saved to {args.save}')


if __name__ == '__main__':
    main()
