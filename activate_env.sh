#!/bin/bash
# ������ ��� ��������� ������������ ���������
# �������������: source activate_env.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/venv/bin/activate"
echo "? ����������� ��������� ������������!"
echo "?? Python: $(python --version)"
echo "?? Pip: $(pip --version | cut -d' ' -f1-2)"
echo ""
echo "��� ����������� ���������: deactivate"


