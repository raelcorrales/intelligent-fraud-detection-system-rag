from typing import Dict, Any

class Transaction:
    """
    Domain entity representing a bank transaction with comprehensive contextual attributes.
    Provides utility methods to serialize details into an embedding-optimized string
    and standard dictionary structures.
    """
    def __init__(self, data: Dict[str, Any]):
        self.TransactionID: str = data.get('TransactionID', 'N/A')
        self.AccountID: str = data.get('AccountID', 'N/A')
        self.TransactionAmount: float = float(data.get('TransactionAmount', 0.0))
        self.TransactionDate: str = data.get('TransactionDate', 'N/A')
        self.TransactionType: str = data.get('TransactionType', 'N/A')
        self.Location: str = data.get('Location', 'N/A')
        self.DeviceID: str = data.get('DeviceID', 'N/A')
        self.IPAddress: str = data.get('IP Address', data.get('IPAddress', 'N/A'))
        self.MerchantID: str = data.get('MerchantID', 'N/A')
        self.Channel: str = data.get('Channel', 'N/A')
        self.CustomerAge: int = int(data.get('CustomerAge', 0))
        self.CustomerOccupation: str = data.get('CustomerOccupation', 'N/A')
        self.TransactionDuration: int = int(data.get('TransactionDuration', 0))
        self.LoginAttempts: int = int(data.get('LoginAttempts', 0))
        self.AccountBalance: float = float(data.get('AccountBalance', 0.0))
        self.PreviousTransactionDate: str = data.get('PreviousTransactionDate', 'N/A')

    def to_string_for_embedding(self) -> str:
        """Converts transaction details into a structured string representation for embeddings and LLM inference."""
        return (
            f"Transaction ID: {self.TransactionID}, Account ID: {self.AccountID}, Amount: ${self.TransactionAmount:.2f}, "
            f"Date: {self.TransactionDate}, Type: {self.TransactionType}, Location: {self.Location}, "
            f"Device: {self.DeviceID}, IP: {self.IPAddress}, Merchant ID: {self.MerchantID}, "
            f"Channel: {self.Channel}, Customer Age: {self.CustomerAge}, Occupation: {self.CustomerOccupation}, "
            f"Transaction Duration: {self.TransactionDuration}s, Login Attempts: {self.LoginAttempts}, "
            f"Account Balance: ${self.AccountBalance:.2f}, Previous Transaction Date: {self.PreviousTransactionDate}."
            f" Context: Transaction type {self.TransactionType} in {self.Location}, via channel {self.Channel}. "
            f"Occupation: {self.CustomerOccupation}, Age: {self.CustomerAge}."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts transaction attributes to a standard dictionary."""
        return {
            'TransactionID': self.TransactionID,
            'AccountID': self.AccountID,
            'TransactionAmount': self.TransactionAmount,
            'TransactionDate': self.TransactionDate,
            'TransactionType': self.TransactionType,
            'Location': self.Location,
            'DeviceID': self.DeviceID,
            'IP Address': self.IPAddress,
            'MerchantID': self.MerchantID,
            'Channel': self.Channel,
            'CustomerAge': self.CustomerAge,
            'CustomerOccupation': self.CustomerOccupation,
            'TransactionDuration': self.TransactionDuration,
            'LoginAttempts': self.LoginAttempts,
            'AccountBalance': self.AccountBalance,
            'PreviousTransactionDate': self.PreviousTransactionDate
        }
