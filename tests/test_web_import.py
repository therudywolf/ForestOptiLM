# SPDX-License-Identifier: AGPL-3.0-or-later
"""Импортеры (web_import.py): разбор ссылок/заголовков. Чистые функции — без сети
(сетевые import_wikipedia/import_gitlab_repo проверяются e2e)."""
from __future__ import annotations

import unittest

import web_import as wi


class TestWikiTitleFromUrl(unittest.TestCase):
    def test_bare_title(self) -> None:
        self.assertEqual(wi._wiki_title_from_url("OWASP"), ("OWASP", ""))

    def test_full_url_lang_and_underscores(self) -> None:
        title, lang = wi._wiki_title_from_url(
            "https://ru.wikipedia.org/wiki/Машинное_обучение")
        self.assertEqual(title, "Машинное обучение")
        self.assertEqual(lang, "ru")

    def test_en_url(self) -> None:
        title, lang = wi._wiki_title_from_url(
            "https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
        self.assertEqual(title, "Retrieval-augmented generation")
        self.assertEqual(lang, "en")

    def test_percent_encoded(self) -> None:
        title, _ = wi._wiki_title_from_url(
            "https://ru.wikipedia.org/wiki/%D0%9E%D0%A1")
        self.assertEqual(title, "ОС")

    def test_canonical_host_no_bogus_lang(self) -> None:
        # https://wikipedia.org/... не должен давать lang="wikipedia" (→ битый хост)
        for url in ("https://wikipedia.org/wiki/Foo",
                    "https://www.wikipedia.org/wiki/Foo",
                    "https://m.wikipedia.org/wiki/Foo"):
            title, lang = wi._wiki_title_from_url(url)
            self.assertEqual(lang, "", url)      # → caller подставит дефолт
            self.assertEqual(title, "Foo")
        # реальный языковой поддомен сохраняется
        self.assertEqual(wi._wiki_title_from_url("https://ru.m.wikipedia.org/wiki/Foo")[1], "ru")


class TestGitlabProjectId(unittest.TestCase):
    def test_full_url_nested_group(self) -> None:
        pid, host = wi._gitlab_project_id("https://gitlab.com/gitlab-org/api/client-go")
        self.assertEqual(pid, "gitlab-org%2Fapi%2Fclient-go")
        self.assertEqual(host, "gitlab.com")

    def test_bare_path(self) -> None:
        pid, host = wi._gitlab_project_id("group/project")
        self.assertEqual(pid, "group%2Fproject")
        self.assertEqual(host, "gitlab.com")

    def test_strips_git_suffix_and_slash(self) -> None:
        pid, _ = wi._gitlab_project_id("https://gitlab.com/g/p.git/")
        self.assertEqual(pid, "g%2Fp")

    def test_self_hosted_host(self) -> None:
        _pid, host = wi._gitlab_project_id("https://git.example.org/team/repo")
        self.assertEqual(host, "git.example.org")


class TestImportedDoc(unittest.TestCase):
    def test_fields(self) -> None:
        d = wi.ImportedDoc(name="n", text="t", origin="o")
        self.assertEqual((d.name, d.text, d.origin), ("n", "t", "o"))


if __name__ == "__main__":
    unittest.main()
