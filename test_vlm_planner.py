#!/usr/bin/env python3
"""
Test VLM-based Task Planner for TypeTele

This script tests the multi-step task planning capabilities
as described in Section 3.2 of the TypeTele paper.

Usage:
  export OPENAI_API_KEY="your_key"
  python test_vlm_planner.py "pick up the bottle and pour water into the cup"
  python test_vlm_planner.py -v  # Interactive mode with verbose output
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from planner.task_planner import TaskPlanner, TaskPlan
from planner.task_executor import TaskExecutor, ExecutionState


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test VLM-based multi-step task planning for TypeTele"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Task description to plan (e.g., 'pick up the bottle and pour water')"
    )
    parser.add_argument(
        "--category",
        default="leap",
        help="Type library category (default: leap)"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("BIGMODEL_API_KEY", ""),
        help="API key for VLM (defaults to BIGMODEL_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        default="https://open.bigmodel.cn/api/paas/v4/",
        help="Base URL for VLM API"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("PLANNER_MODEL", "glm-4.6v"),
        help="VLM model to use (default: glm-4.6v)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision input"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulate task execution (no hardware)"
    )
    args = parser.parse_args()

    if not args.api_key.strip():
        print(
            "Missing API key. Set BIGMODEL_API_KEY environment variable, "
            "or pass --api-key.\n"
            "Example: export BIGMODEL_API_KEY='your_key'",
            file=sys.stderr
        )
        sys.exit(1)

    # Initialize planner
    print(f"Initializing Task Planner with model: {args.model}")
    planner = TaskPlanner(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        category=args.category,
        enable_vision=not args.no_vision
    )
    planner.load_type_library()

    if not planner.type_files:
        print(
            "No gesture types loaded. Add TypeLibrary/<category>/_type_info.json "
            "or at least one .txt file.",
            file=sys.stderr
        )
        sys.exit(1)

    print(f"Loaded {len(planner.type_files)} manipulation types")
    if args.verbose:
        print("\nAvailable types:")
        for type_id in planner.type_files[:10]:
            print(f"  - {type_id}")
        if len(planner.type_files) > 10:
            print(f"  ... and {len(planner.type_files) - 10} more")

    # Interactive or single query mode
    if args.query:
        # Single query mode
        test_queries = [args.query]
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("VLM Task Planner Test - Interactive Mode")
        print("="*60)
        print("\nExample queries:")
        print("  - 'pick up the bottle and pour water into the cup'")
        print("  - 'grasp the tape roll, pull out some tape, and tear it'")
        print("  - 'open the shampoo bottle and press the pump'")
        print("  - 'hold the pencil and write on the paper'")
        print("\nEnter 'quit' to exit, 'test' to run default test cases")
        test_queries = None

    def run_planning(query: str) -> None:
        """Run planning for a single query."""
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        start_time = time.time()
        plan = planner.plan_task(query)
        planning_time = time.time() - start_time

        if plan is None:
            print("[Error] Planning failed")
            return

        print(f"\nPlanning completed in {planning_time:.2f}s")
        print(f"Task: {plan.task_description}")
        print(f"Total steps: {plan.total_steps}")
        print(f"Estimated duration: {plan.estimated_duration:.1f}s")

        # Validate plan
        is_valid, errors = planner.validate_plan(plan)
        if not is_valid:
            print("\n[Warning] Plan validation errors:")
            for error in errors:
                print(f"  - {error}")

        print("\nStep-by-step plan:")
        for step in plan.steps:
            print(f"\n  Step {step.step_number}:")
            print(f"    Description: {step.description}")
            print(f"    Manipulation Type: {step.manipulation_type}")
            print(f"    Hand: {step.hand}")
            print(f"    Objects: {', '.join(step.objects_involved) if step.objects_involved else 'None'}")
            print(f"    Interaction: {step.interaction_type}")
            print(f"    Duration: {step.duration_estimate:.1f}s")

        # Simulate execution if requested
        if args.simulate:
            print("\n" + "-"*60)
            print("Simulating task execution...")
            print("-"*60)

            def on_type_change(type_name: str):
                print(f"  [Sim] Changing type to: {type_name}")

            def on_step_complete(step_num: int, step):
                print(f"  [Sim] Step {step_num} completed: {step.description}")

            def on_task_complete(plan: TaskPlan):
                print(f"  [Sim] Task completed: {plan.task_description}")

            executor = TaskExecutor(
                type_change_callback=on_type_change,
                step_complete_callback=on_step_complete,
                task_complete_callback=on_task_complete,
                default_step_timeout=3.0  # Short timeout for simulation
            )

            executor.start_task(plan)

            # Simulate execution loop
            while executor.is_executing():
                state = executor.update()
                time.sleep(0.5)

            print("-"*60)
            print("Simulation complete")

    # Main loop
    default_tests = [
        "pick up the bottle and pour water into the cup",
        "grasp the tape roll and tear a piece",
        "hold the spray bottle and press the trigger",
        "pick up the small cylinder and place it on the table",
    ]

    if test_queries is None:
        # Interactive mode
        while True:
            try:
                query = input("\nquery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if query.lower() == 'test':
                for test_query in default_tests:
                    run_planning(test_query)
                    time.sleep(1)  # Brief pause between tests
                continue

            run_planning(query)
    else:
        # Single query mode
        for query in test_queries:
            run_planning(query)

    print("\nExiting...")


if __name__ == "__main__":
    main()
