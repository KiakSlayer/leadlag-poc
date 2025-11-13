#!/usr/bin/env python3
"""
Example Tuning Scenarios
========================

Demonstrates different parameter tuning workflows for various use cases.

Author: Quantitative Development Team
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command with description"""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Completed: {description}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed: {description}")
        print(f"Error: {e}")
        return False
    return True


def scenario_1_quick_test():
    """Scenario 1: Quick test with minimal parameters"""
    cmd = [
        'python', 'tuner.py',
        '--intervals', '15m',
        '--windows', '60',
        '--z-entries', '2.0',
        '--z-exits', '0.5',
        '--period', '5d',
        '--max-pairs', '2',
        '--output', 'quick_test_results.csv'
    ]
    run_command(cmd, "Quick Test - Single Configuration")


def scenario_2_comprehensive_sweep():
    """Scenario 2: Comprehensive parameter sweep"""
    cmd = [
        'python', 'tuner.py',
        '--intervals', '5m', '15m',
        '--windows', '30', '60', '90',
        '--z-entries', '1.5', '2.0', '2.5',
        '--z-exits', '0.3', '0.5',
        '--period', '7d',
        '--max-pairs', '3',
        '--output', 'comprehensive_sweep.csv'
    ]
    run_command(cmd, "Comprehensive Sweep - Full Parameter Space")


def scenario_3_fine_tuning():
    """Scenario 3: Fine-tuning around optimal region"""
    cmd = [
        'python', 'tuner.py',
        '--intervals', '15m',
        '--windows', '55', '60', '65',
        '--z-entries', '1.8', '1.9', '2.0', '2.1', '2.2',
        '--z-exits', '0.4', '0.5', '0.6',
        '--period', '7d',
        '--output', 'fine_tuning_results.csv'
    ]
    run_command(cmd, "Fine Tuning - Zoom into Optimal Region")


def scenario_4_stability_test():
    """Scenario 4: Test stability across different intervals"""
    cmd = [
        'python', 'tuner.py',
        '--intervals', '5m', '10m', '15m', '30m',
        '--windows', '60',
        '--z-entries', '2.0',
        '--z-exits', '0.5',
        '--period', '7d',
        '--output', 'stability_test_results.csv'
    ]
    run_command(cmd, "Stability Test - Different Time Intervals")


def scenario_5_crypto_only():
    """Scenario 5: Crypto-only pairs"""
    cmd = [
        'python', 'tuner.py',
        '--crypto', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT',
        '--intervals', '5m', '15m',
        '--windows', '30', '60', '90',
        '--z-entries', '1.5', '2.0', '2.5',
        '--z-exits', '0.3', '0.5',
        '--period', '7d',
        '--output', 'crypto_only_results.csv'
    ]
    run_command(cmd, "Crypto-Only Analysis")


def main():
    """Main menu"""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         Parameter Tuning - Example Scenarios                  ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Choose a scenario to run:
    
    1. Quick Test (1 config, ~2 minutes)
    2. Comprehensive Sweep (36 configs, ~30-60 minutes)
    3. Fine Tuning (45 configs, ~45-90 minutes)
    4. Stability Test (4 configs, ~10 minutes)
    5. Crypto-Only Analysis (36 configs, ~30 minutes)
    
    0. Exit
    """)
    
    while True:
        try:
            choice = input("\nEnter choice (0-5): ").strip()
            
            if choice == '0':
                print("\n👋 Goodbye!")
                sys.exit(0)
            elif choice == '1':
                scenario_1_quick_test()
                break
            elif choice == '2':
                scenario_2_comprehensive_sweep()
                break
            elif choice == '3':
                scenario_3_fine_tuning()
                break
            elif choice == '4':
                scenario_4_stability_test()
                break
            elif choice == '5':
                scenario_5_crypto_only()
                break
            else:
                print("❌ Invalid choice. Please enter 0-5.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    
    print(f"\n\n{'='*80}")
    print("✅ Scenario completed!")
    print(f"{'='*80}")
    print("\n📊 Next steps:")
    print("   1. Check the generated CSV file")
    print("   2. Open tuning_analysis.ipynb to visualize results")
    print("   3. Review recommendations and select optimal parameters")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
