import argparse


def generate_preamble(out, state_dim):
    for i in range(state_dim):
        out.write(f"(declare-const X_{i} Real)\n")
    out.write("(declare-const Y_0 Real)\n")
    out.write("(declare-const Y_1 Real)\n")
    out.write("\n")


def generate_limits(out, box_radius):
    out.write("; Input constraints.\n\n")
    for i, r in enumerate(box_radius):
        out.write(f"; Input state {i}.\n")
        out.write(f"(assert (<= X_{i} {r}))\n")
        out.write(f"(assert (>= X_{i} {-r}))\n\n")


def generate_specs(out, args):
    out.write("; Lyapunov condition.\n\n")
    out.write(f"(assert (<= Y_0 {args.c2}))\n")
    out.write(f"(assert (>= Y_0 {args.c1}))\n")
    out.write(f"(assert (>= Y_1 0))\n")


def generate_csv(args):
    fname = f"{args.name}/specs.csv"
    with open(fname, "w") as out:
        print(f"Generating {fname}")
        out.write(f"../../verification/{args.name}/spec.vnnlib\n")
    print(f"Done.")


def main():
    parser = argparse.ArgumentParser(
        prog="VNNLIB Generator",
        description="Generate VNNLIB property file for verification of Lyapunov condition under level set constraint",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="van",
        help="Name of the system. Default is 'van'.",
    )
    parser.add_argument(
        "-r",
        "--box_radius",
        type=float,
        nargs="+",
        help="Radius of the box. A list of state_dim numbers.",
    )
    parser.add_argument(
        "-c1",
        "--c1",
        type=float,
        default=0.0,
        help="Lower limit of the level set. Default is 0.0.",
    )
    parser.add_argument(
        "-c2",
        "--c2",
        type=float,
        default=1.0,
        help="Upper limit of the level set. Default is 0.0.",
    )

    args = parser.parse_args()

    state_dim = len(args.box_radius)

    fname = f"{args.name}/spec.vnnlib"
    with open(fname, "w") as out:
        print(
            f"Generating {fname} for {args.name} with c1={args.c1}, c2={args.c2}, and box_radius={args.box_radius}"
        )
        generate_preamble(out, state_dim)
        generate_limits(out, args.box_radius)
        generate_specs(out, args)
    generate_csv(args)


if __name__ == "__main__":
    main()