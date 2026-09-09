class ApplicationError(Exception):
    """Base application exception."""


class DatabaseError(ApplicationError):
    """Raised when a database operation fails."""


class CustomerNotFoundError(ApplicationError):
    """Raised when a customer does not exist."""