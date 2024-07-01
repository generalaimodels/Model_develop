# Python Built-in Exceptions

| Exception Class | Description | Base Class | Category |
|---|---|---|---|
| `BaseException` | Base class for all exceptions. |  | Base |
| `SystemExit` | Raised by `sys.exit()`. | `BaseException` | System-related |
| `KeyboardInterrupt` | Raised when the user hits interrupt keys (Ctrl+C). | `BaseException` | User Interrupt |
| `GeneratorExit` | Raised when a generator is closed. | `BaseException` | Generator |
| `Exception` | Base class for all built-in exceptions except those inheriting from `BaseException` directly. | `BaseException` | General |
| `StopIteration` | Raised by `next()` to signal the end of an iterator. | `Exception` | Iterator |
| `StopAsyncIteration` | Raised by `__anext__()` to signal the end of an asynchronous iterator. | `Exception` | Asynchronous Iterator |
| `ArithmeticError` | Base class for arithmetic errors. | `Exception` | Arithmetic |
| `FloatingPointError` | Raised when a floating-point operation fails. | `ArithmeticError` | Arithmetic |
| `OverflowError` | Raised when the result of an arithmetic operation is too large. | `ArithmeticError` | Arithmetic |
| `ZeroDivisionError` | Raised when dividing by zero. | `ArithmeticError` | Arithmetic |
| `AssertionError` | Raised when an `assert` statement fails. | `Exception` | Assertion |
| `AttributeError` | Raised when an attribute reference or assignment fails. | `Exception` | Attribute |
| `BufferError` | Raised when a buffer related operation cannot be performed. | `Exception` | Buffer |
| `EOFError` | Raised when `input()` is called without an available line. | `Exception` | Input/Output |
| `ImportError` | Raised when an import statement fails. | `Exception` | Import |
| `ModuleNotFoundError` | Raised when `import` cannot find the module. | `ImportError` | Import |
| `LookupError` | Base class for lookup errors. | `Exception` | Lookup |
| `IndexError` | Raised when a sequence index is out of range. | `LookupError` | Lookup |
| `KeyError` | Raised when a dictionary key is not found. | `LookupError` | Lookup |
| `MemoryError` | Raised when an operation runs out of memory. | `Exception` | Memory |
| `NameError` | Raised when a local or global name is not found. | `Exception` | Name |
| `UnboundLocalError` | Raised when a local variable is referenced before assignment. | `NameError` | Name |
| `OSError` | Base class for operating system errors. | `Exception` | Operating System |
| `BlockingIOError` | Raised when an operation would block on an object set for non-blocking operation. | `OSError` | I/O |
| `ChildProcessError` | Raised when an operation on a child process fails. | `OSError` | Process |
| `ConnectionError` | Base class for connection related errors. | `OSError` | Network |
| `BrokenPipeError` | Raised when trying to write to a pipe while the other end has been closed. | `ConnectionError` | Network |
| `ConnectionAbortedError` | Raised when a connection attempt is aborted by the peer. | `ConnectionError` | Network |
| `ConnectionRefusedError` | Raised when a connection attempt is refused by the peer. | `ConnectionError` | Network |
| `ConnectionResetError` | Raised when a connection is reset by the peer. | `ConnectionError` | Network |
| `FileExistsError` | Raised when trying to create a file or directory that already exists. | `OSError` | File System |
| `FileNotFoundError` | Raised when trying to open a file that does not exist. | `OSError` | File System |
| `InterruptedError` | Raised when a system call is interrupted by an incoming signal. | `OSError` | System Call |
| `IsADirectoryError` | Raised when trying to open a directory for reading as a file. | `OSError` | File System |
| `NotADirectoryError` | Raised when trying to open something that isn't a directory as a directory. | `OSError` | File System |
| `PermissionError` | Raised when trying to perform an operation without sufficient permissions. | `OSError` | Permissions |
| `ProcessLookupError` | Raised when a given process doesn't exist. | `OSError` | Process |
| `TimeoutError` | Raised when a system function timed out. | `OSError` | Timeout |
| `ReferenceError` | Raised when a weak reference proxy is used to access an object that has already been garbage collected. | `Exception` | Reference |
| `RuntimeError` | Raised when an error doesn't fall under any other category. | `Exception` | Runtime |
| `NotImplementedError` | Raised when an abstract method is not implemented. | `RuntimeError` | Implementation |
| `RecursionError` | Raised when the maximum recursion depth is exceeded. | `RuntimeError` | Recursion |
| `SyntaxError` | Raised when the parser encounters a syntax error. | `Exception` | Syntax |
| `IndentationError` | Raised when indentation is not correct. | `SyntaxError` | Syntax |
| `TabError` | Raised when indentation consists of inconsistent tabs and spaces. | `IndentationError` | Syntax |
| `SystemError` | Raised when the interpreter finds an internal error. | `Exception` | System |
| `TypeError` | Raised when an operation or function is applied to an object of inappropriate type. | `Exception` | Type |
| `ValueError` | Raised when an operation or function receives an argument of the correct type but an inappropriate value. | `Exception` | Value |
| `UnicodeError` | Base class for Unicode related errors. | `ValueError` | Unicode |
| `UnicodeDecodeError` | Raised when a Unicode decoding error occurs. | `UnicodeError` | Unicode |
| `UnicodeEncodeError` | Raised when a Unicode encoding error occurs. | `UnicodeError` | Unicode |
| `UnicodeTranslateError` | Raised when a Unicode translation error occurs. | `UnicodeError` | Unicode |
| `Warning` | Base class for warning categories. | `Exception` | Warning |
| `DeprecationWarning` | Raised when a deprecated feature is used. | `Warning` | Warning |
| `PendingDeprecationWarning` | Raised when a feature will be deprecated in the future. | `Warning` | Warning |
| `RuntimeWarning` | Raised when a suspicious runtime behavior is detected. | `Warning` | Warning |
| `SyntaxWarning` | Raised when a syntax error is detected but the code can still be parsed. | `Warning` | Warning |
| `UserWarning` | Raised when a user code generates a warning. | `Warning` | Warning |
| `FutureWarning` | Raised when a feature is used that will change behavior in the future. | `Warning` | Warning |
| `ImportWarning` | Raised when there is a problem importing a module. | `Warning` | Warning |
| `UnicodeWarning` | Raised when a Unicode related issue occurs. | `Warning` | Warning |
| `BytesWarning` | Raised when there is an issue related to bytes and bytearray operations. | `Warning` | Warning |
| `ResourceWarning` | Raised when a resource is not closed properly. | `Warning` | Warning | 

