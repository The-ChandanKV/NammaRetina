import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LandingApp } from "@/LandingApp";
import { DoctorSignInLink } from "@/components/landing/DoctorSignInLink";
import { site } from "@/lib/site";

describe("Flask landing integration", () => {
  it("renders real document-navigation links to the Flask sign-in page", () => {
    const html = renderToStaticMarkup(<DoctorSignInLink>Doctor Sign In</DoctorSignInLink>);
    expect(html).toContain('href="/signin"');
    expect(html).toContain("Doctor Sign In");
  });
  it("renders the app with valid navigation targets and sign-in actions", () => {
    const html = renderToStaticMarkup(<LandingApp />);
    for (const section of site.sections) {
      expect(html).toContain(`id="${section.id}"`);
      expect(html).toContain(`href="#${section.id}"`);
    }
    expect(html).toContain('href="/signin"');
    expect(html).not.toContain('href="/login"');
    expect(html).toContain("NammaRetina");
  });
});
