import type { AnchorHTMLAttributes, ReactNode } from "react";

import { site } from "@/lib/site";

type Props = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> & {
  children: ReactNode;
};

/**
 * Full-document navigation to Flask `/login`.
 * Must NOT use React Router — otherwise the SPA would intercept `/login`.
 */
export function DoctorSignInLink({ children, ...rest }: Props) {
  return (
    <a href={site.loginPath} {...rest}>
      {children}
    </a>
  );
}
