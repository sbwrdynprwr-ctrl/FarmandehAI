from pathlib import Path

HTML = Path(__file__).resolve().parents[1] / "web" / "index.html"


def test_command_center_is_persian_rtl_and_has_hybrid_scene():
    page = HTML.read_text(encoding="utf-8")
    assert 'lang="fa" dir="rtl"' in page
    assert 'id="world"' in page
    assert "three.module.js" in page
    assert "OrbitControls" in page


def test_ui_explicitly_discloses_simulation_and_unconnected_backend():
    page = HTML.read_text(encoding="utf-8")
    assert "به کارگزار یا صرافی متصل نیست" in page
    assert "هنوز به موتور معاملاتی Python وصل نشده‌اند" in page
    assert "داده‌های بازار و موجودی این نمایش، آزمایشی هستند" in page


def test_live_orders_remain_disabled_and_simulation_controls_exist():
    page = HTML.read_text(encoding="utf-8")
    assert "Live Trading" in page
    assert "خاموش" in page
    assert 'id="runBtn"' in page
    assert "هیچ سفارش واقعی ارسال نمی‌شود" in page
