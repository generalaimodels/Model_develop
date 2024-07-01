# Module Information

## Classes

### Any
Documentation: Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

#### Methods

##### __new__
Documentation: Create and return a new object.  See help(type) for accurate signature.

Arguments: (cls, *args, **kwargs)


## Functions

### __getattr__
Documentation: None

Arguments: (name: str) -> Any

### _warn_on_import
Documentation: Warn on import of deprecated module.

Arguments: (name: str, replacement: Optional[str] = None) -> None

### surface_langchain_deprecation_warnings
Documentation: Unmute LangChain deprecation warnings.

Arguments: () -> None

