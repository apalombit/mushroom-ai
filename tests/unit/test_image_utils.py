"""Unit tests for ingestion.sources._image_utils."""

from bs4 import BeautifulSoup, Tag

from ingestion.sources._image_utils import filter_content_images


def _make_img(src: str, **attrs) -> Tag:
    """Create a BeautifulSoup <img> tag."""
    soup = BeautifulSoup("", "html.parser")
    tag = soup.new_tag("img", src=src, **attrs)
    return tag


def test_resolves_relative_urls():
    imgs = [_make_img("/images/photo.jpg")]
    result = filter_content_images(imgs, "https://example.com/species/page.html")
    assert result == ["https://example.com/images/photo.jpg"]


def test_filters_svg():
    imgs = [_make_img("diagram.svg"), _make_img("photo.jpg")]
    result = filter_content_images(imgs, "https://example.com/")
    assert result == ["https://example.com/photo.jpg"]


def test_filters_data_uri():
    imgs = [_make_img("data:image/png;base64,abc"), _make_img("photo.jpg")]
    result = filter_content_images(imgs, "https://example.com/")
    assert result == ["https://example.com/photo.jpg"]


def test_filters_icons_and_logos():
    imgs = [
        _make_img("site-logo.png"),
        _make_img("nav-icon.gif"),
        _make_img("mushroom.jpg"),
    ]
    result = filter_content_images(imgs, "https://example.com/")
    assert result == ["https://example.com/mushroom.jpg"]


def test_filters_tiny_images():
    imgs = [
        _make_img("small.jpg", width="40", height="40"),
        _make_img("large.jpg", width="400", height="300"),
    ]
    result = filter_content_images(imgs, "https://example.com/")
    assert result == ["https://example.com/large.jpg"]


def test_max_images_cap():
    imgs = [_make_img(f"photo{i}.jpg") for i in range(10)]
    result = filter_content_images(imgs, "https://example.com/", max_images=3)
    assert len(result) == 3


def test_deduplicates_urls():
    imgs = [_make_img("photo.jpg"), _make_img("photo.jpg"), _make_img("other.jpg")]
    result = filter_content_images(imgs, "https://example.com/")
    assert result == ["https://example.com/photo.jpg", "https://example.com/other.jpg"]


def test_empty_input():
    assert filter_content_images([], "https://example.com/") == []


def test_no_src_attribute():
    soup = BeautifulSoup("", "html.parser")
    tag = soup.new_tag("img")  # no src
    result = filter_content_images([tag], "https://example.com/")
    assert result == []
