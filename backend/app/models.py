from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Business(Base):
    __tablename__ = "businesses"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String(120), nullable=False
    )

    sector: Mapped[str] = mapped_column(
        String(40), nullable=False
    )

    province: Mapped[str] = mapped_column(
        String(60), nullable=False
    )
