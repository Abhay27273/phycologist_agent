"""add_google_sub_to_users

Revision ID: 6af97edc5541
Revises: a1b2c3d4e5f6
Create Date: 2026-08-11 21:30:23.816099

Trimmed by hand from --autogenerate's output: it also proposed dropping the
checkpoints/checkpoint_writes/checkpoint_blobs/checkpoint_migrations tables
(LangGraph's own, managed by AsyncPostgresSaver.setup(), not tracked by our
SQLAlchemy models — autogenerate reads that as "should not exist") and
renaming several unrelated pre-existing indexes. Neither belongs in a
migration whose actual job is adding one column.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '6af97edc5541'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('google_sub', sa.String(), nullable=True))
    op.create_index(op.f('ix_users_google_sub'), 'users', ['google_sub'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_users_google_sub'), table_name='users')
    op.drop_column('users', 'google_sub')
