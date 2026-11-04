import unittest

import qaagent


class PackageTests(unittest.TestCase):
    def test_package_is_importable(self) -> None:
        self.assertTrue(qaagent.__file__)


if __name__ == "__main__":
    unittest.main()
