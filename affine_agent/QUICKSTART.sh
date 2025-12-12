#!/bin/bash
# Quick Reference Guide for Dataset Generation
# Run this script to see all available commands

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║         Affine Agent - Dataset Generation Quick Reference             ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 1. TEST THE IMPLEMENTATION"
echo "   Test single task to verify setup:"
echo "   $ python test_dataset_generation.py"
echo ""

echo "🚀 2. GENERATE SMALL SAMPLE (10 tasks)"
echo "   DED tasks:"
echo "   $ python cli.py generate-dataset --env ded --start-id 20000 --end-id 20009 --verbose"
echo ""
echo "   ABD tasks:"
echo "   $ python cli.py generate-dataset --env abd --start-id 20000 --end-id 20009 --verbose"
echo ""

echo "🏭 3. GENERATE FULL DATASETS (20000-23302)"
echo "   Using convenience script:"
echo "   $ bash generate_datasets.sh"
echo ""
echo "   Or manually:"
echo "   $ python cli.py generate-dataset --env abd --start-id 20000 --end-id 23302 --save-interval 50 --verbose"
echo "   $ python cli.py generate-dataset --env ded --start-id 20000 --end-id 23302 --save-interval 50 --verbose"
echo ""

echo "📊 4. OUTPUT FILES"
echo "   Location: datasets/"
echo "   - abd_dataset_YYYYMMDD_HHMMSS.json   (Successful ABD conversations)"
echo "   - ded_dataset_YYYYMMDD_HHMMSS.json   (Successful DED conversations)"
echo "   - abd_stats_YYYYMMDD_HHMMSS.json     (Statistics and metadata)"
echo "   - ded_stats_YYYYMMDD_HHMMSS.json     (Statistics and metadata)"
echo ""

echo "⚙️  5. CONFIGURATION OPTIONS"
echo "   --env {abd,ded}          Environment type (SAT not supported)"
echo "   --start-id N             Starting task ID (inclusive)"
echo "   --end-id N               Ending task ID (inclusive)"
echo "   --save-interval N        Save every N tasks (default: 10)"
echo "   --output-dir DIR         Output directory (default: datasets)"
echo "   --model MODEL            Override model (default: gpt-4o)"
echo "   --temperature T          Override temperature (default: 0.7)"
echo "   --verbose                Enable detailed progress output"
echo "   --api-key KEY            Override API key"
echo ""

echo "📖 6. DOCUMENTATION"
echo "   Full guide:        cat DATASET_GENERATION.md"
echo "   Implementation:    cat DATASET_IMPLEMENTATION.md"
echo "   Main README:       cat README.md"
echo ""

echo "🔍 7. MONITOR PROGRESS"
echo "   Watch stats file (while generation is running):"
echo "   $ watch -n 5 'cat datasets/abd_stats_*.json | jq .'"
echo ""

echo "✅ REQUIREMENTS"
echo "   • OPENAI_API_KEY environment variable set"
echo "   • Python packages: openai, aiohttp, tqdm"
echo "   • Network access to API endpoint"
echo ""

echo "════════════════════════════════════════════════════════════════════════"
