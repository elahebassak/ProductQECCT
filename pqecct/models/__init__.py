from .product_qecct import ProductQECCT
from .qecct import QECCT
from .cross_qecct import CrossQECCT

MODELS = ("product", "qecct", "crossqecct")


def build_model(args, code):
    """args: argparse namespace with model/d_model/h/N_dec/err_types/... fields."""
    if args.model == "product":
        return ProductQECCT(code, d_model=args.d_model, h=args.h, N_dec=args.N_dec,
                            share_processes=not args.no_share, d_g=args.d_g, N_g=args.N_g,
                            no_g=bool(args.no_g), dropout=args.dropout, out_mix=not args.no_out_mix)
    if args.model == "qecct":
        return QECCT(code, err_types=args.err_types, d_model=args.d_model, h=args.h, N_dec=args.N_dec,
                     g_hidden_mult=args.g_hidden_mult, no_g=bool(args.no_g), no_mask=bool(args.no_mask),
                     dropout=args.dropout)
    if args.model == "crossqecct":
        return CrossQECCT(code, err_types=args.err_types, d_model=args.d_model, h=args.h, N_dec=args.N_dec,
                          g_hidden_mult=args.g_hidden_mult, no_g=bool(args.no_g), dropout=args.dropout)
    raise ValueError(args.model)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
