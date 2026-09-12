class ApplicationError(Exception):
    """Base application exception."""

    status_code = 500
    error_code = "INTERNAL_ERROR"


class DatabaseError(ApplicationError):
    """Raised when a database operation fails."""

    status_code = 503
    error_code = "DEPENDENCY_UNAVAILABLE"


class CustomerNotFoundError(ApplicationError):
    """Raised when a customer does not exist."""

    status_code = 404
    error_code = "NOT_FOUND"


class ValidationFailedError(ApplicationError):
    status_code = 400
    error_code = "VALIDATION_ERROR"


class AuthenticationError(ApplicationError):
    status_code = 401
    error_code = "AUTHENTICATION_ERROR"


class AuthorizationError(ApplicationError):
    status_code = 403
    error_code = "AUTHORIZATION_ERROR"


class NotFoundError(ApplicationError):
    status_code = 404
    error_code = "NOT_FOUND"


class RateLimitError(ApplicationError):
    status_code = 429
    error_code = "RATE_LIMIT"


class DependencyUnavailableError(ApplicationError):
    status_code = 503
    error_code = "DEPENDENCY_UNAVAILABLE"


class InternalError(ApplicationError):
    status_code = 500
    error_code = "INTERNAL_ERROR"
