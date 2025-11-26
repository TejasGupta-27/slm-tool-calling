import trl
print(f"TRL version: {trl.__version__}")
try:
    from trl import DataCollatorForCompletionOnlyLM
    print("Import successful from trl")
except ImportError:
    print("Import failed from trl")
    try:
        from trl.trainer import DataCollatorForCompletionOnlyLM
        print("Import successful from trl.trainer")
    except ImportError:
        print("Import failed from trl.trainer too")

