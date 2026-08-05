"""Dependency-free mutation audit for MultiTrain's production modules.

This script mutates only a disposable copy under the operating-system temp
directory. It never edits the working source tree.
"""

import argparse
import ast
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SOURCE_FILES = (
    Path("MultiTrain/utils/utils.py"),
    Path("MultiTrain/classification/classification_models.py"),
    Path("MultiTrain/regression/regression_models.py"),
)

TEST_FILES = (
    "MultiTrain/test/exhaustive_test.py",
    "MultiTrain/test/combinatorial_test.py",
)


@dataclass(frozen=True)
class Mutation:
    source: str
    line: int
    column: int
    node_type: str
    action: str

    @property
    def label(self):
        return f"{self.source}:{self.line}:{self.column} {self.action}"


COMPARE_REPLACEMENTS = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}


class MutationCollector(ast.NodeVisitor):
    def __init__(self, source):
        self.source = source
        self.mutations = []

    def add(self, node, action):
        self.mutations.append(
            Mutation(
                source=str(self.source).replace("\\", "/"),
                line=node.lineno,
                column=node.col_offset,
                node_type=type(node).__name__,
                action=action,
            )
        )

    def visit_If(self, node):
        self.add(node, "negate-if-condition")
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.add(node, "negate-inline-condition")
        self.generic_visit(node)

    def visit_Compare(self, node):
        if len(node.ops) == 1 and type(node.ops[0]) in COMPARE_REPLACEMENTS:
            self.add(node, "invert-comparison")
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.add(node, "swap-and-or")
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            self.add(node, "remove-not")
        self.generic_visit(node)

    def visit_Return(self, node):
        if node.value is not None:
            self.add(node, "return-none")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            self.add(node, "flip-boolean")
        elif isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            self.add(node, "increment-number")


class SingleMutator(ast.NodeTransformer):
    def __init__(self, mutation):
        self.mutation = mutation
        self.applied = False

    def matches(self, node):
        return (
            not self.applied
            and type(node).__name__ == self.mutation.node_type
            and node.lineno == self.mutation.line
            and node.col_offset == self.mutation.column
        )

    def visit_If(self, node):
        self.generic_visit(node)
        if self.matches(node) and self.mutation.action == "negate-if-condition":
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self.applied = True
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        if self.matches(node) and self.mutation.action == "negate-inline-condition":
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self.applied = True
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.matches(node) and self.mutation.action == "invert-comparison":
            replacement = COMPARE_REPLACEMENTS[type(node.ops[0])]
            node.ops[0] = replacement()
            self.applied = True
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self.matches(node) and self.mutation.action == "swap-and-or":
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            self.applied = True
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if self.matches(node) and self.mutation.action == "remove-not":
            self.applied = True
            return node.operand
        return node

    def visit_Return(self, node):
        self.generic_visit(node)
        if self.matches(node) and self.mutation.action == "return-none":
            node.value = ast.Constant(value=None)
            self.applied = True
        return node

    def visit_Constant(self, node):
        if self.matches(node):
            if self.mutation.action == "flip-boolean":
                node.value = not node.value
            elif self.mutation.action == "increment-number":
                node.value += 1
            self.applied = True
        return node


def collect_mutations(project_root):
    mutations = []
    for relative_path in SOURCE_FILES:
        tree = ast.parse((project_root / relative_path).read_text(encoding="utf-8"))
        collector = MutationCollector(relative_path)
        collector.visit(tree)
        mutations.extend(collector.mutations)
    return mutations


def sample_evenly(mutations, limit):
    if limit is None or limit >= len(mutations):
        return mutations
    if limit == 1:
        return [mutations[0]]
    indexes = {
        round(index * (len(mutations) - 1) / (limit - 1)) for index in range(limit)
    }
    return [mutations[index] for index in sorted(indexes)]


def mutate_source(source_text, mutation):
    tree = ast.parse(source_text)
    mutator = SingleMutator(mutation)
    mutated_tree = mutator.visit(tree)
    if not mutator.applied:
        raise RuntimeError(f"Mutation was not applied: {mutation.label}")
    ast.fix_missing_locations(mutated_tree)
    return ast.unparse(mutated_tree) + "\n"


def run_tests(work_root, timeout):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(work_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-x",
        "--disable-warnings",
        *TEST_FILES,
    ]
    return subprocess.run(
        command,
        cwd=work_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    all_mutations = collect_mutations(project_root)
    mutations = sample_evenly(all_mutations, args.limit)
    outcomes = []

    with tempfile.TemporaryDirectory(prefix="multitrain-mutation-") as temp_dir:
        work_root = Path(temp_dir)
        shutil.copytree(project_root / "MultiTrain", work_root / "MultiTrain")

        baseline = run_tests(work_root, args.timeout)
        if baseline.returncode != 0:
            sys.stderr.write(baseline.stdout + baseline.stderr)
            raise SystemExit("Baseline tests failed in the disposable copy")

        print(
            f"Baseline passed. Running {len(mutations)} of {len(all_mutations)} "
            "mechanically generated mutants.",
            flush=True,
        )

        for index, mutation in enumerate(mutations, start=1):
            source_path = work_root / mutation.source
            original = source_path.read_text(encoding="utf-8")
            source_path.write_text(
                mutate_source(original, mutation), encoding="utf-8"
            )
            try:
                result = run_tests(work_root, args.timeout)
                status = "survived" if result.returncode == 0 else "killed"
            except subprocess.TimeoutExpired:
                status = "timeout-killed"
            finally:
                source_path.write_text(original, encoding="utf-8")

            outcomes.append({**asdict(mutation), "label": mutation.label, "status": status})
            if index % 5 == 0 or status == "survived":
                print(
                    f"[{index}/{len(mutations)}] {status}: {mutation.label}",
                    flush=True,
                )

    killed = sum(item["status"] != "survived" for item in outcomes)
    survived = len(outcomes) - killed
    score = 100.0 * killed / len(outcomes) if outcomes else 0.0
    report = {
        "generated_mutations": len(all_mutations),
        "sampled_mutations": len(outcomes),
        "killed": killed,
        "survived": survived,
        "mutation_score": score,
        "outcomes": outcomes,
    }
    report_path = project_root / ".pytest_cache" / "mutation-report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Mutation score: {score:.1f}% ({killed} killed, {survived} survived).",
        flush=True,
    )
    print(f"Report: {report_path}", flush=True)
    if survived:
        print("Survivors:", flush=True)
        for item in outcomes:
            if item["status"] == "survived":
                print(f"- {item['label']}", flush=True)


if __name__ == "__main__":
    main()
